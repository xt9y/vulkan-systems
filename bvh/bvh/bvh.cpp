#include "bvh.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace bvh
{
    namespace
    {
        state* data=nullptr;
        struct parameters
        {
            vec4 world_min,world_max;
            std::array<std::uint32_t,4> sizes,sort;
        };
        static_assert(sizeof(parameters)==64);

        bool finite(float value){return std::isfinite(value);}

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
                      std::uint32_t count,const parameters& params)
        {
            if(!count)return;
            vk::record::pipeline(command,vk::PipelineBindPoint::eCompute,
                                 data->pipelines[stage]);
            command.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                       data->pipeline_layout,0,{data->descriptors},{});
            command.pushConstants(data->pipeline_layout,
                                  vk::ShaderStageFlagBits::eCompute,
                                  0,sizeof(params),&params);
            vk::record::dispatch(command,(count+127u)/128u,1,1);
            barrier(command);
        }

        vk::buffer::resource allocate(std::size_t bytes,bool cpu_write=false)
        {
            return vk::buffer::create(
                bytes,vk::BufferUsageFlagBits::eStorageBuffer,
                cpu_write?(vk::MemoryPropertyFlagBits::eHostVisible |
                           vk::MemoryPropertyFlagBits::eHostCoherent):
                          vk::MemoryPropertyFlagBits::eDeviceLocal);
        }

        parameters args()
        {
            parameters result{};
            result.world_min=data->world_min;
            result.world_max=data->world_max;
            result.sizes={data->primitive_count,data->padded_count,
                          0,data->max_hits_per_query};
            return result;
        }
    }

    state* get(){return data;}

    void init(const std::vector<aabb>& boxes,const shader_paths& shaders,
              std::uint32_t max_queries,std::uint32_t max_hits_per_query)
    {
        if(data)throw std::logic_error("BVH already initialized");
        if(boxes.empty() || boxes.size()>65536u ||
           max_queries>65536u || max_hits_per_query>1024u ||
           !max_hits_per_query || !vk::device::get()->handle)
            throw std::invalid_argument("Invalid BVH capacity or device");
        data=new state{};
        data->primitive_count=std::uint32_t(boxes.size());
        data->max_queries=max_queries;
        data->max_hits_per_query=max_hits_per_query;
        std::uint32_t padded=1u;
        while(padded<data->primitive_count)padded*=2u;
        data->padded_count=padded;
        vec4 minimum={std::numeric_limits<float>::max(),
                      std::numeric_limits<float>::max(),
                      std::numeric_limits<float>::max(),0.f};
        vec4 maximum={-minimum.x,-minimum.y,-minimum.z,0.f};
        for(const auto& box:boxes)
        {
            for(int k=0;k<3;++k)
            {
                const float low=(&box.minimum.x)[k];
                const float high=(&box.maximum.x)[k];
                if(!finite(low)||!finite(high)||low>high)
                {
                    delete data;
                    data=nullptr;
                    throw std::invalid_argument("Invalid BVH primitive AABB");
                }
                (&minimum.x)[k]=std::min((&minimum.x)[k],low);
                (&maximum.x)[k]=std::max((&maximum.x)[k],high);
            }
        }
        data->world_min=minimum;
        data->world_max=maximum;
        const auto device=vk::device::get()->handle;
        try
        {
            data->primitives=allocate(boxes.size()*sizeof(aabb),true);
            vk::buffer::write(data->primitives,boxes.data(),
                              boxes.size()*sizeof(aabb));
            data->keys=allocate(std::size_t(padded)*sizeof(std::uint32_t)*2u);
            data->nodes=allocate((boxes.size()*2u-1u)*sizeof(node));
            data->arrivals=allocate(std::max<std::size_t>(
                1u,boxes.size()-1u)*sizeof(std::uint32_t));
            const std::size_t queries=std::max<std::size_t>(1u,max_queries);
            data->queries=allocate(queries*sizeof(aabb),true);
            data->hit_counts=allocate(queries*sizeof(std::uint32_t));
            data->hit_indices=allocate(queries*max_hits_per_query*
                                      sizeof(std::uint32_t));
            data->rays=allocate(queries*sizeof(ray),true);
            data->ray_hits=allocate(queries*sizeof(ray_hit));

            std::array<vk::DescriptorSetLayoutBinding,9> bindings{};
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
            data->layout=vk::descriptor_layout::create(layout);
            vk::PushConstantRange range{};
            range.stageFlags=vk::ShaderStageFlagBits::eCompute;
            range.size=sizeof(parameters);
            vk::PipelineLayoutCreateInfo pipeline_layout{};
            pipeline_layout.setLayoutCount=1;
            pipeline_layout.pSetLayouts=&data->layout;
            pipeline_layout.pushConstantRangeCount=1;
            pipeline_layout.pPushConstantRanges=&range;
            data->pipeline_layout=vk::pipeline_layout::create(pipeline_layout);

            for(std::uint32_t i=0;i<data->pipelines.size();++i)
            {
                auto shader=vk::shader::create(vk::shader::read(shaders[i]));
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

            vk::DescriptorPoolSize pool_size{
                vk::DescriptorType::eStorageBuffer,9
            };
            vk::DescriptorPoolCreateInfo pool{};
            pool.maxSets=1;
            pool.poolSizeCount=1;
            pool.pPoolSizes=&pool_size;
            data->pool=device.createDescriptorPool(pool);
            vk::DescriptorSetAllocateInfo allocation{};
            allocation.descriptorPool=data->pool;
            allocation.descriptorSetCount=1;
            allocation.pSetLayouts=&data->layout;
            data->descriptors=device.allocateDescriptorSets(allocation).front();

            std::array<vk::DescriptorBufferInfo,9> infos{{
                vk::buffer::descriptor(data->primitives),
                vk::buffer::descriptor(data->keys),
                vk::buffer::descriptor(data->nodes),
                vk::buffer::descriptor(data->arrivals),
                vk::buffer::descriptor(data->queries),
                vk::buffer::descriptor(data->hit_counts),
                vk::buffer::descriptor(data->hit_indices),
                vk::buffer::descriptor(data->rays),
                vk::buffer::descriptor(data->ray_hits)
            }};
            std::array<vk::WriteDescriptorSet,9> writes{};
            for(std::uint32_t i=0;i<writes.size();++i)
            {
                writes[i].dstSet=data->descriptors;
                writes[i].dstBinding=i;
                writes[i].descriptorCount=1;
                writes[i].descriptorType=vk::DescriptorType::eStorageBuffer;
                writes[i].pBufferInfo=&infos[i];
            }
            device.updateDescriptorSets(writes,{});
        }
        catch(...)
        {
            shutdown();
            throw;
        }
    }

    void record_build(vk::CommandBuffer command)
    {
        if(!data || !command)
            throw std::logic_error("BVH build called before initialization");
        auto push=args();
        dispatch(command,0,data->padded_count,push); // Morton codes
        // Portable, subgroup-independent bitonic sort baseline.
        // Only the tree topology and refit run in O(N) GPU work.
        for(std::uint32_t k=2;k<=data->padded_count;k*=2u)
        {
            for(std::uint32_t j=k/2u;j>0u;j/=2u)
            {
                push.sort={j,k,0,0};
                dispatch(command,1,data->padded_count,push);
            }
        }
        push.sort={0,0,0,0};
        dispatch(command,2,data->primitive_count*2u-1u,push);
        dispatch(command,3,data->primitive_count-1u,push);
        dispatch(command,4,data->primitive_count,push);
        dispatch(command,5,data->primitive_count,push);
    }

    void record_overlaps(vk::CommandBuffer command,std::uint32_t count)
    {
        if(!data || count>data->max_queries)
            throw std::invalid_argument("BVH overlap query capacity exceeded");
        auto push=args();
        push.sizes[2]=count;
        dispatch(command,6,count,push);
    }

    void record_rays(vk::CommandBuffer command,std::uint32_t count)
    {
        if(!data || count>data->max_queries)
            throw std::invalid_argument("BVH ray query capacity exceeded");
        auto push=args();
        push.sizes[2]=count;
        dispatch(command,7,count,push);
    }

    vk::DescriptorBufferInfo nodes_descriptor()
    {
        if(!data)throw std::logic_error("BVH not initialized");
        return vk::buffer::descriptor(data->nodes);
    }
    vk::DescriptorBufferInfo queries_descriptor()
    {
        if(!data)throw std::logic_error("BVH not initialized");
        return vk::buffer::descriptor(data->queries);
    }
    vk::DescriptorBufferInfo hits_descriptor()
    {
        if(!data)throw std::logic_error("BVH not initialized");
        return vk::buffer::descriptor(data->hit_indices);
    }
    vk::DescriptorBufferInfo counts_descriptor()
    {
        if(!data)throw std::logic_error("BVH not initialized");
        return vk::buffer::descriptor(data->hit_counts);
    }
    vk::DescriptorBufferInfo rays_descriptor()
    {
        if(!data)throw std::logic_error("BVH not initialized");
        return vk::buffer::descriptor(data->rays);
    }
    vk::DescriptorBufferInfo ray_hits_descriptor()
    {
        if(!data)throw std::logic_error("BVH not initialized");
        return vk::buffer::descriptor(data->ray_hits);
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
            for(auto* resource:{
                &data->primitives,&data->keys,&data->nodes,&data->arrivals,
                &data->queries,&data->hit_counts,&data->hit_indices,
                &data->rays,&data->ray_hits})
                if(resource->handle)vk::buffer::destroy(*resource);
        }
        delete data;
        data=nullptr;
    }
}
