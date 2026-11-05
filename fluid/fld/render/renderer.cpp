#include "renderer.hpp"
#include "../core/system.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace fld::render
{
    namespace
    {
        state *data=nullptr;
        struct parameters
        {
            mat4 transform;
            vec4 absorption,scattering,lighting;
            std::array<std::uint32_t,4> sizes;
        };
        static_assert(sizeof(parameters)==128);

        mat4 inverse(const mat4 &m)
        {
            double a[4][8]{};
            for(int r=0;r<4;++r)
            {
                for(int c=0;c<4;++c)a[r][c]=m(r,c);
                a[r][r+4]=1.;
            }
            for(int c=0;c<4;++c)
            {
                int pivot=c;
                for(int r=c+1;r<4;++r)
                    if(std::abs(a[r][c])>std::abs(a[pivot][c]))pivot=r;
                if(!std::isfinite(a[pivot][c]) ||
                   std::abs(a[pivot][c])<1e-12)
                    throw std::invalid_argument("Singular fluid camera");
                for(int k=0;k<8;++k)std::swap(a[c][k],a[pivot][k]);
                const auto divisor=a[c][c];
                for(int k=0;k<8;++k)a[c][k]/=divisor;
                for(int r=0;r<4;++r)
                {
                    if(r==c)continue;
                    const auto q=a[r][c];
                    for(int k=0;k<8;++k)a[r][k]-=q*a[c][k];
                }
            }
            mat4 result{};
            for(int r=0;r<4;++r)
                for(int c=0;c<4;++c)result(r,c)=float(a[r][c+4]);
            return result;
        }

        void barrier(vk::CommandBuffer cmd)
        {
            vk::MemoryBarrier memory{};
            memory.srcAccessMask=vk::AccessFlagBits::eShaderWrite;
            memory.dstAccessMask=vk::AccessFlagBits::eShaderRead |
                                 vk::AccessFlagBits::eShaderWrite;
            cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                vk::PipelineStageFlagBits::eComputeShader,
                                {},{memory},{},{});
        }

        void dispatch(vk::CommandBuffer cmd,std::uint32_t stage,
                      const frame &f,const parameters &push,
                      std::uint32_t x,std::uint32_t y)
        {
            vk::record::pipeline(cmd,vk::PipelineBindPoint::eCompute,
                                 data->pipelines[stage]);
            cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                   data->pipeline_layout,0,{f.descriptors},{});
            cmd.pushConstants(data->pipeline_layout,
                              vk::ShaderStageFlagBits::eCompute,
                              0,sizeof(push),&push);
            vk::record::dispatch(cmd,x,y,1);
            barrier(cmd);
        }
    }

    state *get(){return data;}

    bool active()
    {
        return data && system::particle_count()>0;
    }

    void init(const std::array<std::string,3> &paths,
              std::uint32_t count,std::uint32_t width,std::uint32_t height)
    {
        if(data)throw std::logic_error("Fluid renderer initialized twice");
        if(!system::get() || !count || !width || !height ||
           width>8192 || height>8192 ||
           !vk::device::get()->handle)
            throw std::invalid_argument("Invalid fluid renderer state");
        data=new state{};
        data->width=width;
        data->height=height;
        auto device=vk::device::get()->handle;
        try
        {
            std::array<vk::DescriptorSetLayoutBinding,6> bindings{};
            for(std::uint32_t i=0;i<bindings.size();++i)
            {
                bindings[i].binding=i;
                bindings[i].descriptorCount=1;
                bindings[i].stageFlags=vk::ShaderStageFlagBits::eCompute;
                bindings[i].descriptorType=i<2 ?
                    vk::DescriptorType::eCombinedImageSampler:
                    i==2 ? vk::DescriptorType::eStorageImage:
                           vk::DescriptorType::eStorageBuffer;
            }
            vk::DescriptorSetLayoutCreateInfo layout{};
            layout.bindingCount=bindings.size();
            layout.pBindings=bindings.data();
            data->layout=vk::descriptor_layout::create(layout);

            vk::PushConstantRange range{};
            range.stageFlags=vk::ShaderStageFlagBits::eCompute;
            range.size=sizeof(parameters);
            vk::PipelineLayoutCreateInfo info{};
            info.setLayoutCount=1;
            info.pSetLayouts=&data->layout;
            info.pushConstantRangeCount=1;
            info.pPushConstantRanges=&range;
            data->pipeline_layout=vk::pipeline_layout::create(info);

            for(std::uint32_t i=0;i<data->pipelines.size();++i)
            {
                auto shader=vk::shader::create(vk::shader::read(paths[i]));
                try
                {
                    vk::ComputePipelineCreateInfo pipeline{};
                    pipeline.stage.stage=vk::ShaderStageFlagBits::eCompute;
                    pipeline.stage.module=shader;
                    pipeline.stage.pName="main";
                    pipeline.layout=data->pipeline_layout;
                    data->pipelines[i]=vk::pipeline::compute(pipeline);
                }
                catch(...)
                {
                    vk::shader::destroy(shader);
                    throw;
                }
                vk::shader::destroy(shader);
            }

            std::array<vk::DescriptorPoolSize,3> sizes{{
                {vk::DescriptorType::eCombinedImageSampler,count*2},
                {vk::DescriptorType::eStorageImage,count},
                {vk::DescriptorType::eStorageBuffer,count*3}
            }};
            vk::DescriptorPoolCreateInfo pool{};
            pool.maxSets=count;
            pool.poolSizeCount=sizes.size();
            pool.pPoolSizes=sizes.data();
            data->pool=device.createDescriptorPool(pool);

            std::vector<vk::DescriptorSetLayout> layouts(count,data->layout);
            vk::DescriptorSetAllocateInfo allocation{};
            allocation.descriptorPool=data->pool;
            allocation.descriptorSetCount=count;
            allocation.pSetLayouts=layouts.data();
            const auto sets=device.allocateDescriptorSets(allocation);
            data->frames.resize(count);

            const std::size_t tiles=std::size_t((width+15)/16)*
                                    std::size_t((height+15)/16);
            for(std::uint32_t i=0;i<count;++i)
            {
                auto &f=data->frames[i];
                f.descriptors=sets[i];
                // Entirely GPU-updated bins; no per-frame CPU tile copies.
                const auto usage=vk::BufferUsageFlagBits::eStorageBuffer;
                const auto memory=vk::MemoryPropertyFlagBits::eDeviceLocal;
                f.tile_counts=vk::buffer::create(tiles*sizeof(std::uint32_t),
                                                 usage,memory);
                f.tile_indices=vk::buffer::create(tiles*2048u*sizeof(std::uint32_t),
                                                  usage,memory);
            }
        }
        catch(...)
        {
            shutdown();
            throw;
        }
    }

    void record(vk::CommandBuffer command,std::uint32_t image,
                const input &target)
    {
        if(!active() || !command || image>=data->frames.size() ||
           !target.depth || !target.scene_color || !target.output ||
           !target.sampler || target.width!=data->width ||
           target.height!=data->height)
            throw std::invalid_argument("Invalid fluid render input");

        auto &f=data->frames[image];
        const std::array<vk::DescriptorImageInfo,3> images{{
            {target.sampler,target.depth,target.depth_layout},
            {target.sampler,target.scene_color,target.color_layout},
            {{},target.output,vk::ImageLayout::eGeneral}
        }};
        const std::array<vk::DescriptorBufferInfo,3> buffers{{
            system::descriptor(),
            vk::buffer::descriptor(f.tile_counts),
            vk::buffer::descriptor(f.tile_indices)
        }};
        std::array<vk::WriteDescriptorSet,6> writes{};
        for(std::uint32_t i=0;i<writes.size();++i)
        {
            writes[i].dstSet=f.descriptors;
            writes[i].dstBinding=i;
            writes[i].descriptorCount=1;
            if(i<3)
            {
                writes[i].descriptorType=i==2 ?
                    vk::DescriptorType::eStorageImage:
                    vk::DescriptorType::eCombinedImageSampler;
                writes[i].pImageInfo=&images[i];
            }
            else
            {
                writes[i].descriptorType=vk::DescriptorType::eStorageBuffer;
                writes[i].pBufferInfo=&buffers[i-3];
            }
        }
        vk::device::get()->handle.updateDescriptorSets(writes,{});

        const auto *world=system::get();
        parameters push{};
        push.transform=target.view_projection;
        push.absorption={.27f,.10f,.045f,
                         world->settings.liquid.index_of_refraction};
        push.scattering={.018f,.036f,.050f,.065f};
        push.lighting={.42f,.85f,.3f,world->settings.spacing};
        const std::uint32_t ratio=std::uint32_t(std::clamp(
            std::round(world->settings.smoothing_radius/
                       world->settings.spacing*100.f),
            150.f,65535.f));
        push.sizes={target.width,target.height,
                    (target.reversed_depth?1u:0u)|(ratio<<8),
                    world->count};

        const auto tiles=((target.width+15)/16)*((target.height+15)/16);
        dispatch(command,0,f,push,(tiles+63)/64,1);      // clear tile bins
        dispatch(command,1,f,push,(world->count+63)/64,1); // scatter particles
        push.transform=inverse(target.view_projection);
        dispatch(command,2,f,push,(target.width+7)/8,
                 (target.height+7)/8);                  // surface+optics
    }

    void shutdown()
    {
        if(!data)return;
        const auto device=vk::device::get()->handle;
        if(device)
        {
            for(auto &f:data->frames)
            {
                if(f.tile_counts.handle)vk::buffer::destroy(f.tile_counts);
                if(f.tile_indices.handle)vk::buffer::destroy(f.tile_indices);
            }
            for(auto &pipeline:data->pipelines)
                if(pipeline)vk::pipeline::destroy(pipeline);
            if(data->pipeline_layout)
                vk::pipeline_layout::destroy(data->pipeline_layout);
            if(data->pool)device.destroyDescriptorPool(data->pool);
            if(data->layout)vk::descriptor_layout::destroy(data->layout);
        }
        delete data;
        data=nullptr;
    }
}
