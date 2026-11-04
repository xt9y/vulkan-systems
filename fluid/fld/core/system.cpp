#include "system.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace fld::system
{
    namespace
    {
        state *data=nullptr;
        constexpr float pi=3.14159265358979323846f;
        struct parameters
        {
            vec4 gravity_dt;
            vec4 liquid;
            std::array<std::uint32_t,4> indices;
        };
        static_assert(sizeof(parameters)==48);

        void barrier(vk::CommandBuffer command)
        {
            vk::MemoryBarrier memory{};
            memory.srcAccessMask=vk::AccessFlagBits::eShaderWrite;
            memory.dstAccessMask=vk::AccessFlagBits::eShaderRead |
                                 vk::AccessFlagBits::eShaderWrite;
            command.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                    vk::PipelineStageFlagBits::eComputeShader,
                                    {},{memory},{},{});
        }

        void dispatch(vk::CommandBuffer command,std::uint32_t stage,
                      std::uint32_t count,const parameters &push)
        {
            vk::record::pipeline(command,vk::PipelineBindPoint::eCompute,
                                 data->pipelines[stage]);
            command.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                       data->pipeline_layout,0,{data->descriptors},{});
            command.pushConstants(data->pipeline_layout,
                                  vk::ShaderStageFlagBits::eCompute,
                                  0,sizeof(push),&push);
            vk::record::dispatch(command,(count+63)/64,1,1);
            barrier(command);
        }
    }

    state *get(){return data;}

    std::uint32_t particle_count()
    {
        return data?data->count:0;
    }

    vk::DescriptorBufferInfo descriptor()
    {
        if(!data)throw std::logic_error("Fluid GPU particles not initialized");
        return vk::buffer::descriptor(data->particles);
    }

    void init(const options &settings,const drop &initial,
              const shader_paths &paths)
    {
        if(data)throw std::logic_error("VulkanFluid already initialized");
        if(!valid(settings.liquid) || !settings.max_particles ||
           settings.max_particles>65536 || settings.pressure_iterations>12 ||
           !settings.pressure_iterations || settings.substeps>8 ||
           !settings.substeps || settings.spacing<=0.f ||
           settings.smoothing_radius<settings.spacing*1.8f ||
           initial.radius<=settings.spacing ||
           !vk::device::get()->handle)
            throw std::invalid_argument("Invalid VulkanFluid GPU settings");

        std::vector<particle> seeded;
        const auto spacing=settings.spacing;
        const int reach=int(std::ceil(initial.radius/spacing));
        const float volume=spacing*spacing*spacing;
        for(int z=-reach;z<=reach;++z)
            for(int y=-reach;y<=reach;++y)
                for(int x=-reach;x<=reach;++x)
                {
                    const vec3 local{float(x)*spacing,float(y)*spacing,
                                     float(z)*spacing};
                    if(local.x*local.x+local.y*local.y+local.z*local.z>
                       initial.radius*initial.radius)continue;
                    if(seeded.size()>=settings.max_particles)
                        throw std::runtime_error(
                            "Drop exceeds particle capacity; reduce radius or spacing");
                    particle item{};
                    item.position={initial.center.x+local.x,
                                   initial.center.y+local.y,
                                   initial.center.z+local.z,spacing*.54f};
                    item.velocity={initial.velocity.x,initial.velocity.y,
                                   initial.velocity.z,0.f};
                    item.previous=item.position;
                    item.physical={volume,settings.liquid.density*volume,0.f,0.f};
                    seeded.push_back(item);
                }
        if(seeded.empty())throw std::invalid_argument("Empty initial liquid");
        data=new state{};
        data->settings=settings;
        data->count=std::uint32_t(seeded.size());
        auto device=vk::device::get()->handle;
        try
        {
            const auto usage=vk::BufferUsageFlagBits::eStorageBuffer;
            const auto memory=vk::MemoryPropertyFlagBits::eHostVisible |
                              vk::MemoryPropertyFlagBits::eHostCoherent;
            data->particles=vk::buffer::create(seeded.size()*sizeof(particle),
                                                usage,memory);
            data->heads=vk::buffer::create(data->bucket_count*sizeof(std::int32_t),
                                            usage,memory);
            data->next=vk::buffer::create(seeded.size()*sizeof(std::int32_t),
                                           usage,memory);
            data->lambdas=vk::buffer::create(seeded.size()*sizeof(float),
                                              usage,memory);
            data->deltas=vk::buffer::create(seeded.size()*sizeof(vec4),
                                             usage,memory);
            vk::buffer::write(data->particles,seeded.data(),
                              seeded.size()*sizeof(particle));

            std::array<vk::DescriptorSetLayoutBinding,5> bindings{};
            for(std::uint32_t i=0;i<bindings.size();++i)
            {
                bindings[i].binding=i;
                bindings[i].descriptorCount=1;
                bindings[i].descriptorType=vk::DescriptorType::eStorageBuffer;
                bindings[i].stageFlags=vk::ShaderStageFlagBits::eCompute;
            }
            vk::DescriptorSetLayoutCreateInfo layout{};
            layout.bindingCount=bindings.size();
            layout.pBindings=bindings.data();
            data->descriptor_layout=vk::descriptor_layout::create(layout);

            vk::PushConstantRange range{};
            range.stageFlags=vk::ShaderStageFlagBits::eCompute;
            range.size=sizeof(parameters);
            vk::PipelineLayoutCreateInfo info{};
            info.setLayoutCount=1;
            info.pSetLayouts=&data->descriptor_layout;
            info.pushConstantRangeCount=1;
            info.pPushConstantRanges=&range;
            data->pipeline_layout=vk::pipeline_layout::create(info);

            for(std::uint32_t i=0;i<data->pipelines.size();++i)
            {
                const auto code=vk::shader::read(paths[i]);
                auto shader=vk::shader::create(code);
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

            vk::DescriptorPoolSize size{
                vk::DescriptorType::eStorageBuffer,5
            };
            vk::DescriptorPoolCreateInfo pool{};
            pool.maxSets=1;
            pool.poolSizeCount=1;
            pool.pPoolSizes=&size;
            data->descriptor_pool=device.createDescriptorPool(pool);
            vk::DescriptorSetAllocateInfo allocation{};
            allocation.descriptorPool=data->descriptor_pool;
            allocation.descriptorSetCount=1;
            allocation.pSetLayouts=&data->descriptor_layout;
            data->descriptors=device.allocateDescriptorSets(allocation).front();

            const std::array<vk::DescriptorBufferInfo,5> buffers{{
                vk::buffer::descriptor(data->particles),
                vk::buffer::descriptor(data->heads),
                vk::buffer::descriptor(data->next),
                vk::buffer::descriptor(data->lambdas),
                vk::buffer::descriptor(data->deltas)
            }};
            std::array<vk::WriteDescriptorSet,5> writes{};
            for(std::uint32_t i=0;i<writes.size();++i)
            {
                writes[i].dstSet=data->descriptors;
                writes[i].dstBinding=i;
                writes[i].descriptorCount=1;
                writes[i].descriptorType=vk::DescriptorType::eStorageBuffer;
                writes[i].pBufferInfo=&buffers[i];
            }
            device.updateDescriptorSets(writes,{});
        }
        catch(...)
        {
            shutdown();
            throw;
        }
    }

    void record(vk::CommandBuffer command,float seconds)
    {
        if(!data || !command)throw std::logic_error("Invalid fluid command");
        if(!std::isfinite(seconds) || seconds<0.f)
            throw std::invalid_argument("Invalid elapsed fluid time");
        if(seconds==0.f)return;
        // Render and physics rates are independent. Large frame stalls cannot
        // drive an unbounded GPU solve; record the simulated time separately.
        const float dt=std::min(seconds,1.f/30.f)/
                       float(data->settings.substeps);
        parameters push{};
        push.gravity_dt={data->settings.gravity.x,data->settings.gravity.y,
                         data->settings.gravity.z,dt};
        push.liquid={data->settings.liquid.density,
                     data->settings.spacing,
                     data->settings.smoothing_radius,
                     data->settings.floor_height};
        push.indices={data->count,data->bucket_count,
                      data->settings.pressure_iterations,0};
        barrier(command); // RAW dependency from earlier queued submissions
        for(std::uint32_t sub=0;sub<data->settings.substeps;++sub)
        {
            dispatch(command,1,data->count,push);         // predict
            dispatch(command,0,data->bucket_count,push); // clear
            dispatch(command,2,data->count,push);         // build spatial hash
            for(std::uint32_t i=0;i<data->settings.pressure_iterations;++i)
            {
                dispatch(command,3,data->count,push);     // constraints
                dispatch(command,4,data->count,push);     // Jacobi correction
                dispatch(command,5,data->count,push);     // apply
            }
            dispatch(command,6,data->count,push);         // final velocity
        }
        data->simulated_seconds+=dt*data->settings.substeps;
    }

    void shutdown()
    {
        if(!data)return;
        const auto device=vk::device::get()->handle;
        if(device)
        {
            for(auto &pipeline:data->pipelines)
                if(pipeline)vk::pipeline::destroy(pipeline);
            if(data->pipeline_layout)
                vk::pipeline_layout::destroy(data->pipeline_layout);
            if(data->descriptor_pool)
                device.destroyDescriptorPool(data->descriptor_pool);
            if(data->descriptor_layout)
                vk::descriptor_layout::destroy(data->descriptor_layout);
            if(data->particles.handle)vk::buffer::destroy(data->particles);
            if(data->heads.handle)vk::buffer::destroy(data->heads);
            if(data->next.handle)vk::buffer::destroy(data->next);
            if(data->lambdas.handle)vk::buffer::destroy(data->lambdas);
            if(data->deltas.handle)vk::buffer::destroy(data->deltas);
        }
        delete data;
        data=nullptr;
    }
}
