#include "renderer.hpp"
#include "../core/system.hpp"
#include "../math/kernels.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace fld::render
{
    namespace
    {
        state *data = nullptr;
        constexpr std::uint32_t tile_size = 16;

        struct gpu_particle
        {
            vec4 world;
            vec4 properties; // x volume m^3, y mass kg, z material ID
        };
        static_assert(sizeof(gpu_particle) == 32);

        struct parameters
        {
            mat4 inverse_view_projection;
            vec4 absorption;
            vec4 scattering;
            vec4 lighting;
            std::array<std::uint32_t,4> sizes{};
        };
        static_assert(sizeof(parameters) == 128);

        // Variable-length, compact per-tile lists: dense splashes must not
        // lose fluid simply because more than 96 particles overlap a tile.
        struct bins
        {
            std::uint32_t columns, rows;
            std::uint32_t peak=0;
            std::vector<std::vector<std::uint32_t>> entries;
            std::vector<std::array<std::uint32_t,2>> ranges;
            std::vector<std::uint32_t> indices;

            bins(std::uint32_t width, std::uint32_t height)
            {
                columns=(width+tile_size-1)/tile_size;
                rows=(height+tile_size-1)/tile_size;
                const auto count=std::size_t(columns)*rows;
                entries.resize(count);
                ranges.resize(count);
            }

            void add(std::uint32_t id,float left,float top,
                     float right,float bottom)
            {
                if(right<0 || bottom<0 ||
                   left>=float(columns*tile_size) ||
                   top>=float(rows*tile_size))return;
                const auto lo_x=std::clamp(int(std::floor(left/tile_size)),0,int(columns)-1);
                const auto hi_x=std::clamp(int(std::floor(right/tile_size)),0,int(columns)-1);
                const auto lo_y=std::clamp(int(std::floor(top/tile_size)),0,int(rows)-1);
                const auto hi_y=std::clamp(int(std::floor(bottom/tile_size)),0,int(rows)-1);
                for(int y=lo_y;y<=hi_y;++y)
                    for(int x=lo_x;x<=hi_x;++x)
                    {
                        auto &tile=entries[std::size_t(y)*columns+x];
                        tile.push_back(id);
                        peak=std::max(peak,std::uint32_t(tile.size()));
                    }
            }

            void pack()
            {
                std::size_t total=0;
                constexpr auto limit=std::numeric_limits<std::uint32_t>::max();
                for(const auto &tile:entries)
                {
                    if(tile.size()>limit-total)
                        throw std::overflow_error("Fluid tile index count exceeds 32 bits");
                    total+=tile.size();
                }
                indices.reserve(total);
                for(std::size_t i=0;i<entries.size();++i)
                {
                    const auto &tile=entries[i];
                    ranges[i]={std::uint32_t(indices.size()),
                               std::uint32_t(tile.size())};
                    indices.insert(indices.end(),tile.begin(),tile.end());
                }
            }
        };

        void make_buffers(frame &f, std::uint32_t width, std::uint32_t height,
                          std::uint32_t maximum)
        {
            const auto memory = vk::MemoryPropertyFlagBits::eHostVisible |
                                vk::MemoryPropertyFlagBits::eHostCoherent;
            const auto usage = vk::BufferUsageFlagBits::eStorageBuffer;
            const auto tiles = std::size_t((width+15)/16)*((height+15)/16);
            f.particles=vk::buffer::create(
                std::max<std::size_t>(1,maximum)*sizeof(gpu_particle),
                usage,memory);
            f.tile_counts=vk::buffer::create(
                std::max<std::size_t>(1,tiles)*sizeof(std::array<std::uint32_t,2>),
                usage,memory);
            f.tile_indices=vk::buffer::create(
                std::max<std::size_t>(1,maximum)*sizeof(std::uint32_t),
                usage,memory);
        }

        void bind(frame &f, const input &target)
        {
            std::array<vk::DescriptorImageInfo,3> images{};
            images[0]={target.sampler,target.depth,target.depth_layout};
            images[1]={target.sampler,target.scene_color,target.color_layout};
            images[2]={{},target.output,vk::ImageLayout::eGeneral};
            std::array<vk::DescriptorBufferInfo,3> buffers{};
            buffers[0]=vk::buffer::descriptor(f.particles);
            buffers[1]=vk::buffer::descriptor(f.tile_counts);
            buffers[2]=vk::buffer::descriptor(f.tile_indices);
            std::array<vk::WriteDescriptorSet,6> writes{};
            for(std::uint32_t i=0;i<writes.size();++i)
            {
                auto &w=writes[i];
                w.dstSet=f.descriptors;
                w.dstBinding=i;
                w.descriptorCount=1;
                if(i<3)
                {
                    w.descriptorType=i==2?vk::DescriptorType::eStorageImage:
                                                   vk::DescriptorType::eCombinedImageSampler;
                    w.pImageInfo=&images[i];
                }
                else
                {
                    w.descriptorType=vk::DescriptorType::eStorageBuffer;
                    w.pBufferInfo=&buffers[i-3];
                }
            }
            vk::device::get()->handle.updateDescriptorSets(writes,{});
        }
    }

    state *get() { return data; }

    bool active()
    {
        const auto *world=system::get();
        return world && !world->particles.empty();
    }

    void init(const std::vector<std::uint32_t> &spirv,
              std::uint32_t frames, std::uint32_t width,
              std::uint32_t height)
    {
        if(data) throw std::logic_error("Fluid renderer already initialized");
        if(!frames || !width || !height || width>8192 || height>8192 ||
           spirv.empty() || !vk::device::get()->handle)
            throw std::invalid_argument("Invalid fluid Vulkan initialization");

        data=new state{};
        data->width=width;
        data->height=height;
        data->max_particles=system::get() ?
            system::get()->settings.max_particles:8192;
        const auto device=vk::device::get()->handle;
        try
        {
            std::array<vk::DescriptorSetLayoutBinding,6> bindings{};
            for(std::uint32_t i=0;i<bindings.size();++i)
            {
                bindings[i].binding=i;
                bindings[i].descriptorCount=1;
                bindings[i].stageFlags=vk::ShaderStageFlagBits::eCompute;
                bindings[i].descriptorType=i<2 ?
                    vk::DescriptorType::eCombinedImageSampler :
                    i==2 ? vk::DescriptorType::eStorageImage :
                           vk::DescriptorType::eStorageBuffer;
            }
            vk::DescriptorSetLayoutCreateInfo info{};
            info.bindingCount=bindings.size();
            info.pBindings=bindings.data();
            data->descriptor_layout=vk::descriptor_layout::create(info);

            vk::PushConstantRange push{};
            push.stageFlags=vk::ShaderStageFlagBits::eCompute;
            push.size=sizeof(parameters);
            vk::PipelineLayoutCreateInfo layout{};
            layout.setLayoutCount=1;
            layout.pSetLayouts=&data->descriptor_layout;
            layout.pushConstantRangeCount=1;
            layout.pPushConstantRanges=&push;
            data->pipeline_layout=vk::pipeline_layout::create(layout);

            auto shader=vk::shader::create(spirv);
            try
            {
                vk::ComputePipelineCreateInfo compute{};
                compute.stage.stage=vk::ShaderStageFlagBits::eCompute;
                compute.stage.module=shader;
                compute.stage.pName="main";
                compute.layout=data->pipeline_layout;
                data->pipeline=vk::pipeline::compute(compute);
            }
            catch (...)
            {
                vk::shader::destroy(shader);
                throw;
            }
            vk::shader::destroy(shader);

            std::array<vk::DescriptorPoolSize,3> sizes{{
                {vk::DescriptorType::eCombinedImageSampler,frames*2},
                {vk::DescriptorType::eStorageImage,frames},
                {vk::DescriptorType::eStorageBuffer,frames*3}
            }};
            vk::DescriptorPoolCreateInfo pool{};
            pool.maxSets=frames;
            pool.poolSizeCount=sizes.size();
            pool.pPoolSizes=sizes.data();
            data->descriptor_pool=device.createDescriptorPool(pool);

            std::vector<vk::DescriptorSetLayout> layouts(
                frames,data->descriptor_layout);
            vk::DescriptorSetAllocateInfo allocation{};
            allocation.descriptorPool=data->descriptor_pool;
            allocation.descriptorSetCount=frames;
            allocation.pSetLayouts=layouts.data();
            const auto sets=device.allocateDescriptorSets(allocation);
            data->frames.resize(frames);
            for(std::uint32_t i=0;i<frames;++i)
            {
                auto &frame=data->frames[i];
                frame.descriptors=sets[i];
                make_buffers(frame,width,height,data->max_particles);
            }
        }
        catch (...)
        {
            shutdown();
            throw;
        }
    }

    void init(const std::string &path, std::uint32_t frames,
              std::uint32_t width,std::uint32_t height)
    {
        init(vk::shader::read(path),frames,width,height);
    }

    void record(vk::CommandBuffer command,std::uint32_t frame_index,
                const input &target)
    {
        if(!data || frame_index>=data->frames.size() || !command ||
           !target.depth || !target.scene_color || !target.output ||
           !target.sampler || !target.width || !target.height ||
           target.width>data->width || target.height>data->height)
            throw std::invalid_argument("Invalid fluid render targets");

        const auto *world=system::get();
        const auto spacing=world ? world->settings.spacing : .004f;
        const auto smoothing=world ? world->settings.smoothing_radius : .010f;
        auto &frame=data->frames[frame_index];
        bins tiles(target.width,target.height);
        std::vector<gpu_particle> points;
        if(world)
        {
            points.reserve(std::min(
                world->particles.size(),std::size_t(data->max_particles)));
            for(const auto &p:world->particles)
            {
                if(target.phase!=0xffffffffu && p.phase!=target.phase)continue;
                if(points.size()>=data->max_particles)break;
                const auto clip=target.view_projection *
                    vec4{p.position.x,p.position.y,p.position.z,1.f};
                if(clip.w<=1e-5f)continue;
                const auto x=(clip.x/clip.w*.5f+.5f)*target.width;
                const auto y=(clip.y/clip.w*.5f+.5f)*target.height;
                const auto z=clip.z/clip.w;
                if(z<0.f || z>1.f)continue;
                const float radius=std::clamp(
                    smoothing*std::abs(target.view_projection(1,1))*
                    target.height*.5f/clip.w,1.f,2048.f);
                const auto id=std::uint32_t(points.size());
                const float volume=p.mass /
                    std::max(.0001f,world->materials[p.phase].density);
                points.push_back({{p.position.x,p.position.y,
                                   p.position.z,p.radius},
                                  {volume,p.mass,float(p.phase),0.f}});
                tiles.add(id,x-radius-1.f,y-radius-1.f,
                          x+radius+1.f,y+radius+1.f);
            }
        }

        tiles.pack();
        data->tile_overflows=0;
        data->peak_tile_occupancy=tiles.peak;
        if(!points.empty())
            vk::buffer::write(frame.particles,points.data(),
                              points.size()*sizeof(gpu_particle));
        vk::buffer::write(frame.tile_counts,tiles.ranges.data(),
                          tiles.ranges.size()*sizeof(std::array<std::uint32_t,2>));
        const auto bytes=tiles.indices.size()*sizeof(std::uint32_t);
        if(frame.tile_indices.size<bytes)
        {
            // Frame resources may be resized only after its fence is signaled,
            // as already required by record() for CPU writes.
            const auto capacity=std::max<std::size_t>(
                bytes,frame.tile_indices.size*2);
            vk::buffer::destroy(frame.tile_indices);
            const auto memory=vk::MemoryPropertyFlagBits::eHostVisible |
                              vk::MemoryPropertyFlagBits::eHostCoherent;
            frame.tile_indices=vk::buffer::create(
                capacity,vk::BufferUsageFlagBits::eStorageBuffer,memory);
        }
        if(bytes)
            vk::buffer::write(frame.tile_indices,tiles.indices.data(),bytes);
        bind(frame,target);

        const appearance *optical=&target.liquid;
        if(target.use_phase_appearance && world &&
           target.phase<world->materials.size())
            optical=&world->materials[target.phase].optical;
        parameters push{};
        push.inverse_view_projection=target.inverse_view_projection;
        push.absorption={optical->absorption.x,
                         optical->absorption.y,
                         optical->absorption.z,
                         optical->index_of_refraction};
        push.scattering={optical->scattering.x,
                         optical->scattering.y,
                         optical->scattering.z,
                         optical->roughness};
        const auto light=math::safe_normal(optical->light_direction);
        push.lighting={light.x,light.y,light.z,spacing};
        const std::uint32_t ratio=std::uint32_t(std::clamp(
            std::round(smoothing/spacing*100.f),100.f,65535.f));
        push.sizes={target.width,target.height,
                    (target.reversed_depth?1u:0u)|(ratio<<8),
                    std::uint32_t(points.size())};

        vk::record::pipeline(command,vk::PipelineBindPoint::eCompute,
                             data->pipeline);
        command.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                   data->pipeline_layout,0,{frame.descriptors},{});
        command.pushConstants(data->pipeline_layout,
                              vk::ShaderStageFlagBits::eCompute,
                              0,sizeof(push),&push);
        vk::record::dispatch(command,
                             (target.width+7)/8,(target.height+7)/8,1);
    }

    void shutdown()
    {
        if(!data)return;
        for(auto &frame:data->frames)
        {
            if(frame.particles.handle)vk::buffer::destroy(frame.particles);
            if(frame.tile_counts.handle)vk::buffer::destroy(frame.tile_counts);
            if(frame.tile_indices.handle)vk::buffer::destroy(frame.tile_indices);
        }
        if(data->pipeline)vk::pipeline::destroy(data->pipeline);
        if(data->pipeline_layout)vk::pipeline_layout::destroy(data->pipeline_layout);
        if(data->descriptor_pool)
            vk::device::get()->handle.destroyDescriptorPool(data->descriptor_pool);
        if(data->descriptor_layout)
            vk::descriptor_layout::destroy(data->descriptor_layout);
        delete data;
        data=nullptr;
    }
}
