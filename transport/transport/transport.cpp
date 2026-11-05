#include "transport.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace transport
{
    namespace
    {
        state* data=nullptr;
        struct parameters
        {
            std::uint32_t patches,blockers,pad0,pad1;
        };

        void check()
        {
            if(!data || !vk::device::get()->handle)
                throw std::logic_error("transport not initialized");
        }

        vk::buffer::resource buffer(std::size_t bytes,vk::BufferUsageFlags usage,
                                    bool upload=false)
        {
            const auto properties=upload?
                (vk::MemoryPropertyFlagBits::eHostVisible |
                 vk::MemoryPropertyFlagBits::eHostCoherent):
                vk::MemoryPropertyFlags(vk::MemoryPropertyFlagBits::eDeviceLocal);
            return vk::buffer::create(bytes,usage,properties);
        }

        void compute_barrier(vk::CommandBuffer command)
        {
            vk::MemoryBarrier memory{};
            memory.srcAccessMask=vk::AccessFlagBits::eShaderWrite;
            memory.dstAccessMask=vk::AccessFlagBits::eShaderRead |
                                 vk::AccessFlagBits::eShaderWrite;
            command.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                    vk::PipelineStageFlagBits::eComputeShader,
                                    {},{memory},{},{});
        }
    }

    state* get(){return data;}

    void init(const std::vector<patch>& surfaces,
              const std::vector<blocker>& obstacles,
              const shader_paths& shaders)
    {
        if(data)throw std::logic_error("transport already initialized");
        if(surfaces.empty() || surfaces.size()>4096 || obstacles.size()>1024 ||
           !vk::device::get()->handle)
            throw std::invalid_argument("invalid transport configuration");

        for(const auto& p:surfaces)
        {
            const float* position=&p.position_area.x;
            const float* normal=&p.normal.x;
            const float* reflectance=&p.albedo.x;
            const float* emission=&p.emission.x;
            if(!std::isfinite(p.position_area.w) || p.position_area.w<=0.f)
                throw std::invalid_argument("patch area must be positive");
            for(int c=0;c<3;c++)
                if(!std::isfinite(position[c]) || !std::isfinite(normal[c]) ||
                   !std::isfinite(reflectance[c]) || !std::isfinite(emission[c]) ||
                   reflectance[c]<0.f || reflectance[c]>=1.f || emission[c]<0.f)
                    throw std::invalid_argument("invalid transport patch");
        }

        data=new state{};
        data->patch_count=std::uint32_t(surfaces.size());
        data->blocker_count=std::uint32_t(obstacles.size());
        const auto device=vk::device::get()->handle;
        try
        {
            const auto n=std::size_t(data->patch_count);
            data->patches=buffer(n*sizeof(patch),vk::BufferUsageFlagBits::eStorageBuffer,true);
            data->blockers=buffer(std::max<std::size_t>(1,obstacles.size())*sizeof(blocker),
                                  vk::BufferUsageFlagBits::eStorageBuffer,true);
            data->exchange=buffer(n*n*sizeof(float),
                                  vk::BufferUsageFlagBits::eStorageBuffer);
            for(auto& b:data->radiance)
                b=buffer(n*sizeof(vec4),vk::BufferUsageFlagBits::eStorageBuffer |
                         vk::BufferUsageFlagBits::eTransferDst);
            vk::buffer::write(data->patches,surfaces.data(),n*sizeof(patch));
            if(!obstacles.empty())
                vk::buffer::write(data->blockers,obstacles.data(),
                                  obstacles.size()*sizeof(blocker));

            std::array<vk::DescriptorSetLayoutBinding,5> bindings{};
            for(std::uint32_t i=0;i<bindings.size();++i)
            {
                bindings[i].binding=i;
                bindings[i].descriptorType=vk::DescriptorType::eStorageBuffer;
                bindings[i].descriptorCount=1;
                bindings[i].stageFlags=vk::ShaderStageFlagBits::eCompute;
            }
            vk::DescriptorSetLayoutCreateInfo layout_info{};
            layout_info.bindingCount=std::uint32_t(bindings.size());
            layout_info.pBindings=bindings.data();
            data->layout=vk::descriptor_layout::create(layout_info);

            vk::PushConstantRange range{};
            range.stageFlags=vk::ShaderStageFlagBits::eCompute;
            range.size=sizeof(parameters);
            vk::PipelineLayoutCreateInfo pipeline_info{};
            pipeline_info.setLayoutCount=1;
            pipeline_info.pSetLayouts=&data->layout;
            pipeline_info.pushConstantRangeCount=1;
            pipeline_info.pPushConstantRanges=&range;
            data->pipeline_layout=vk::pipeline_layout::create(pipeline_info);

            for(std::size_t i=0;i<shaders.size();++i)
            {
                auto shader=vk::shader::create(vk::shader::read(shaders[i]));
                try
                {
                    vk::ComputePipelineCreateInfo compute{};
                    compute.stage.stage=vk::ShaderStageFlagBits::eCompute;
                    compute.stage.module=shader;
                    compute.stage.pName="main";
                    compute.layout=data->pipeline_layout;
                    data->pipelines[i]=vk::pipeline::compute(compute);
                }
                catch(...)
                {
                    vk::shader::destroy(shader);
                    throw;
                }
                vk::shader::destroy(shader);
            }

            vk::DescriptorPoolSize size{vk::DescriptorType::eStorageBuffer,10};
            vk::DescriptorPoolCreateInfo pool{};
            pool.maxSets=2;
            pool.poolSizeCount=1;
            pool.pPoolSizes=&size;
            data->pool=device.createDescriptorPool(pool);

            std::array<vk::DescriptorSetLayout,2> layouts{data->layout,data->layout};
            vk::DescriptorSetAllocateInfo alloc{};
            alloc.descriptorPool=data->pool;
            alloc.descriptorSetCount=2;
            alloc.pSetLayouts=layouts.data();
            auto sets=device.allocateDescriptorSets(alloc);
            data->sets={sets[0],sets[1]};

            for(std::uint32_t s=0;s<2;++s)
            {
                std::array<vk::DescriptorBufferInfo,5> infos{{
                    vk::buffer::descriptor(data->patches),
                    vk::buffer::descriptor(data->exchange),
                    vk::buffer::descriptor(data->radiance[s]),
                    vk::buffer::descriptor(data->radiance[1-s]),
                    vk::buffer::descriptor(data->blockers)
                }};
                std::array<vk::WriteDescriptorSet,5> writes{};
                for(std::uint32_t i=0;i<writes.size();++i)
                {
                    writes[i].dstSet=data->sets[s];
                    writes[i].dstBinding=i;
                    writes[i].descriptorCount=1;
                    writes[i].descriptorType=vk::DescriptorType::eStorageBuffer;
                    writes[i].pBufferInfo=&infos[i];
                }
                device.updateDescriptorSets(writes,{});
            }
        }
        catch(...)
        {
            shutdown();
            throw;
        }
    }

    void record_build(vk::CommandBuffer command)
    {
        check();
        if(!command)throw std::invalid_argument("null command buffer");
        for(auto& r:data->radiance)
            command.fillBuffer(r.handle,0,r.size,0u);
        vk::MemoryBarrier from_transfer{};
        from_transfer.srcAccessMask=vk::AccessFlagBits::eTransferWrite;
        from_transfer.dstAccessMask=vk::AccessFlagBits::eShaderRead |
                                    vk::AccessFlagBits::eShaderWrite;
        command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                vk::PipelineStageFlagBits::eComputeShader,
                                {},{from_transfer},{},{});

        const parameters p{data->patch_count,data->blocker_count,0,0};
        vk::record::pipeline(command,vk::PipelineBindPoint::eCompute,
                             data->pipelines[0]);
        command.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                   data->pipeline_layout,0,{data->sets[0]},{});
        command.pushConstants(data->pipeline_layout,vk::ShaderStageFlagBits::eCompute,
                              0,sizeof(p),&p);
        const auto edges=std::uint64_t(data->patch_count)*data->patch_count;
        vk::record::dispatch(command,std::uint32_t((edges+127)/128),1,1);
        compute_barrier(command);
    }

    void record_solve(vk::CommandBuffer command,std::uint32_t iterations)
    {
        check();
        if(!command || iterations==0 || iterations>256 || (iterations&1u))
            throw std::invalid_argument("transport iterations must be even and <=256");
        const parameters p{data->patch_count,data->blocker_count,0,0};
        vk::record::pipeline(command,vk::PipelineBindPoint::eCompute,
                             data->pipelines[1]);
        command.pushConstants(data->pipeline_layout,vk::ShaderStageFlagBits::eCompute,
                              0,sizeof(p),&p);
        for(std::uint32_t i=0;i<iterations;++i)
        {
            command.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                       data->pipeline_layout,0,{data->sets[i&1u]},{});
            vk::record::dispatch(command,(data->patch_count+127)/128,1,1);
            compute_barrier(command);
        }
    }

    void update_emission(std::uint32_t index,vec4 value)
    {
        check();
        if(index>=data->patch_count)
            throw std::out_of_range("transport patch index");
        const float* channels=&value.x;
        for(int i=0;i<3;++i)
            if(!std::isfinite(channels[i]) || channels[i]<0.f)
                throw std::invalid_argument("invalid emission");
        vk::buffer::write(data->patches,&value,sizeof(value),
                          std::size_t(index)*sizeof(patch)+offsetof(patch,emission));
    }

    vk::DescriptorBufferInfo patches_descriptor()
    {
        check();
        return vk::buffer::descriptor(data->patches);
    }

    vk::DescriptorBufferInfo radiance_descriptor()
    {
        check();
        return vk::buffer::descriptor(data->radiance[0]);
    }

    void shutdown()
    {
        if(!data)return;
        const auto device=vk::device::get()->handle;
        if(device)
        {
            for(auto& pipeline:data->pipelines)
                if(pipeline)vk::pipeline::destroy(pipeline);
            if(data->pipeline_layout)
                vk::pipeline_layout::destroy(data->pipeline_layout);
            if(data->pool)device.destroyDescriptorPool(data->pool);
            if(data->layout)vk::descriptor_layout::destroy(data->layout);
            for(auto* r:{&data->patches,&data->blockers,&data->exchange,
                         &data->radiance[0],&data->radiance[1]})
                if(r->handle)vk::buffer::destroy(*r);
        }
        delete data;
        data=nullptr;
    }
}
