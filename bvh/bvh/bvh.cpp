#include "bvh.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace bvh
{
    // Shader pipelines/layouts are created once for all BLAS/TLAS trees that
    // share the same SPIR-V shader paths and Vulkan device.
    struct shared_pipelines
    {
        vk::Device device{};
        shader_paths paths{};
        vk::DescriptorSetLayout layout{};
        vk::PipelineLayout pipeline_layout{};
        std::array<vk::Pipeline,10> pipelines{};
        ~shared_pipelines() noexcept
        {
            // An already-destroyed device implicitly reclaimed these handles.
            const auto* current=vk::device::get();
            if(!current || current->handle!=device || !device)return;
            try
            {
                for(auto& p:pipelines)
                    if(p)vk::pipeline::destroy(p);
                if(pipeline_layout)vk::pipeline_layout::destroy(pipeline_layout);
                if(layout)vk::descriptor_layout::destroy(layout);
            }
            catch(...) {}
        }
    };

    namespace
    {
        std::mutex program_mutex;
        std::vector<std::weak_ptr<shared_pipelines>> program_cache;

        std::shared_ptr<shared_pipelines> acquire_program(
            const shader_paths& paths,vk::Device device)
        {
            std::lock_guard<std::mutex> lock(program_mutex);
            for(auto it=program_cache.begin();it!=program_cache.end();)
            {
                auto program=it->lock();
                if(!program)
                {
                    it=program_cache.erase(it);
                    continue;
                }
                if(program->device==device && program->paths==paths)
                    return program;
                ++it;
            }

            auto program=std::make_shared<shared_pipelines>();
            program->device=device;
            program->paths=paths;
            std::array<vk::DescriptorSetLayoutBinding,13> bindings{};
            for(std::uint32_t i=0;i<bindings.size();++i)
            {
                auto& binding=bindings[i];
                binding.binding=i;
                binding.descriptorCount=1;
                binding.descriptorType=vk::DescriptorType::eStorageBuffer;
                binding.stageFlags=vk::ShaderStageFlagBits::eCompute;
            }
            vk::DescriptorSetLayoutCreateInfo info{};
            info.bindingCount=bindings.size();
            info.pBindings=bindings.data();
            program->layout=vk::descriptor_layout::create(info);

            vk::PushConstantRange range{};
            range.stageFlags=vk::ShaderStageFlagBits::eCompute;
            range.size=64;
            vk::PipelineLayoutCreateInfo pipeline_layout{};
            pipeline_layout.setLayoutCount=1;
            pipeline_layout.pSetLayouts=&program->layout;
            pipeline_layout.pushConstantRangeCount=1;
            pipeline_layout.pPushConstantRanges=&range;
            program->pipeline_layout=vk::pipeline_layout::create(pipeline_layout);

            for(std::size_t i=0;i<program->pipelines.size();++i)
            {
                auto shader=vk::shader::create(vk::shader::read(paths[i]));
                try
                {
                    vk::ComputePipelineCreateInfo pipeline{};
                    pipeline.stage.stage=vk::ShaderStageFlagBits::eCompute;
                    pipeline.stage.module=shader;
                    pipeline.stage.pName="main";
                    pipeline.layout=program->pipeline_layout;
                    program->pipelines[i]=vk::pipeline::compute(pipeline);
                }
                catch(...)
                {
                    vk::shader::destroy(shader);
                    throw;
                }
                vk::shader::destroy(shader);
            }
            program_cache.push_back(program);
            return program;
        }
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

        void dispatch(state* data,vk::CommandBuffer command,std::uint32_t stage,
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
            vk::MemoryPropertyFlags memory{
                vk::MemoryPropertyFlagBits::eDeviceLocal
            };
            if(cpu_write)
                memory=vk::MemoryPropertyFlagBits::eHostVisible |
                       vk::MemoryPropertyFlagBits::eHostCoherent;
            return vk::buffer::create(
                bytes,vk::BufferUsageFlagBits::eStorageBuffer |
                      vk::BufferUsageFlagBits::eTransferSrc |
                      vk::BufferUsageFlagBits::eTransferDst,memory);
        }

        parameters args(const state* data)
        {
            parameters result{};
            result.world_min=data->world_min;
            result.world_max=data->world_max;
            result.sizes={data->primitive_count,data->padded_count,
                          0,data->max_hits_per_query};
            return result;
        }
    }


    void tree::init_internal(const std::vector<aabb>& boxes,const shader_paths& shaders,
                       std::uint32_t max_queries,std::uint32_t max_hits_per_query,
                       const std::vector<triangle>* geometry)
    {
        if(data_)throw std::logic_error("BVH already initialized");
        if(boxes.empty() || boxes.size()>65536u ||
           max_queries>65536u || max_hits_per_query>1024u ||
           !max_hits_per_query || !vk::device::get()->handle)
            throw std::invalid_argument("Invalid BVH capacity or device");
        data_=new state{};
        data_->triangle_scene=(geometry!=nullptr);
        data_->primitive_count=std::uint32_t(boxes.size());
        data_->max_queries=max_queries;
        data_->max_hits_per_query=max_hits_per_query;
        std::uint32_t padded=1u;
        while(padded<data_->primitive_count)padded*=2u;
        data_->padded_count=padded;
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
                    delete data_;
                    data_=nullptr;
                    throw std::invalid_argument("Invalid BVH primitive AABB");
                }
                (&minimum.x)[k]=std::min((&minimum.x)[k],low);
                (&maximum.x)[k]=std::max((&maximum.x)[k],high);
            }
        }
        data_->world_min=minimum;
        data_->world_max=maximum;
        const auto device=vk::device::get()->handle;
        try
        {
            data_->triangles=allocate((geometry?geometry->size():1u)*sizeof(triangle),true);
            if(geometry)
                vk::buffer::write(data_->triangles,geometry->data(),
                                  geometry->size()*sizeof(triangle));
            data_->primitives=allocate(boxes.size()*sizeof(aabb),true);
            vk::buffer::write(data_->primitives,boxes.data(),
                              boxes.size()*sizeof(aabb));
            data_->keys=allocate(std::size_t(padded)*sizeof(std::uint32_t)*2u);
            data_->sort_scratch=allocate(std::size_t(padded)*sizeof(std::uint32_t)*2u);
            const auto groups=(padded+127u)/128u;
            data_->sort_histogram=allocate(std::size_t(groups)*16u*sizeof(std::uint32_t));
            data_->sort_prefix=allocate(std::size_t(groups)*16u*sizeof(std::uint32_t));
            data_->nodes=allocate((boxes.size()*2u-1u)*sizeof(node));
            data_->arrivals=allocate(std::max<std::size_t>(
                1u,boxes.size()-1u)*sizeof(std::uint32_t));
            const std::size_t queries=std::max<std::size_t>(1u,max_queries);
            data_->queries=allocate(queries*sizeof(aabb),true);
            data_->hit_counts=allocate(queries*sizeof(std::uint32_t));
            data_->hit_indices=allocate(queries*max_hits_per_query*
                                      sizeof(std::uint32_t));
            data_->rays=allocate(queries*sizeof(ray),true);
            data_->ray_hits=allocate(queries*sizeof(ray_hit));

            data_->shared_program=acquire_program(shaders,device);
            data_->layout=data_->shared_program->layout;
            data_->pipeline_layout=data_->shared_program->pipeline_layout;
            data_->pipelines=data_->shared_program->pipelines;

            vk::DescriptorPoolSize pool_size{
                vk::DescriptorType::eStorageBuffer,13
            };
            vk::DescriptorPoolCreateInfo pool{};
            pool.maxSets=1;
            pool.poolSizeCount=1;
            pool.pPoolSizes=&pool_size;
            data_->pool=device.createDescriptorPool(pool);
            vk::DescriptorSetAllocateInfo allocation{};
            allocation.descriptorPool=data_->pool;
            allocation.descriptorSetCount=1;
            allocation.pSetLayouts=&data_->layout;
            data_->descriptors=device.allocateDescriptorSets(allocation).front();

            std::array<vk::DescriptorBufferInfo,13> infos{{
                vk::buffer::descriptor(data_->primitives),
                vk::buffer::descriptor(data_->keys),
                vk::buffer::descriptor(data_->nodes),
                vk::buffer::descriptor(data_->arrivals),
                vk::buffer::descriptor(data_->queries),
                vk::buffer::descriptor(data_->hit_counts),
                vk::buffer::descriptor(data_->hit_indices),
                vk::buffer::descriptor(data_->rays),
                vk::buffer::descriptor(data_->ray_hits),
                vk::buffer::descriptor(data_->triangles),
                vk::buffer::descriptor(data_->sort_scratch),
                vk::buffer::descriptor(data_->sort_histogram),
                vk::buffer::descriptor(data_->sort_prefix)
            }};
            std::array<vk::WriteDescriptorSet,13> writes{};
            for(std::uint32_t i=0;i<writes.size();++i)
            {
                writes[i].dstSet=data_->descriptors;
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

    void tree::init(const std::vector<aabb>& boxes,const shader_paths& shaders,
              std::uint32_t max_queries,std::uint32_t max_hits_per_query)
    {
        init_internal(boxes,shaders,max_queries,max_hits_per_query,nullptr);
    }

    void tree::init_triangles(const std::vector<triangle>& geometry,const shader_paths& shaders,
                        std::uint32_t max_queries,std::uint32_t max_hits_per_query)
    {
        if(geometry.empty() || geometry.size()>65536u)
            throw std::invalid_argument("BVH triangle capacity exceeded");
        std::vector<aabb> bounds;
        bounds.reserve(geometry.size());
        for(const triangle& tri:geometry)
        {
            aabb box{};
            for(int axis=0;axis<3;++axis)
            {
                const float x=(&tri.a.x)[axis], y=(&tri.b.x)[axis],
                            z=(&tri.c.x)[axis];
                if(!finite(x) || !finite(y) || !finite(z))
                    throw std::invalid_argument("Non-finite BVH triangle coordinate");
                (&box.minimum.x)[axis]=std::min({x,y,z});
                (&box.maximum.x)[axis]=std::max({x,y,z});
            }
            bounds.push_back(box);
        }
        init_internal(bounds,shaders,max_queries,max_hits_per_query,&geometry);
    }

    void tree::record_build(vk::CommandBuffer command)
    {
        if(!data_ || !command)
            throw std::logic_error("BVH build called before initialization");
        auto push=args(data_);
        dispatch(data_,command,0,data_->padded_count,push); // Morton codes
        // Stable, 4-bit LSD radix sort: 8 passes, 3 dispatches each.
        // No subgroup assumptions and no quadratic work in dispatch count.
        const std::uint32_t groups=(data_->padded_count+127u)/128u;
        for(std::uint32_t shift=0;shift<32u;shift+=4u)
        {
            push.sort={shift,(shift/4u)&1u,groups,0u};
            dispatch(data_,command,7,data_->padded_count,push);
            dispatch(data_,command,8,16u,push);
            dispatch(data_,command,9,data_->padded_count,push);
        }
        push.sort={0,0,0,0};
        dispatch(data_,command,1,data_->primitive_count*2u-1u,push);
        dispatch(data_,command,2,data_->primitive_count-1u,push);
        dispatch(data_,command,3,data_->primitive_count,push);
        dispatch(data_,command,4,data_->primitive_count,push);
    }

    void tree::update_bounds(const std::vector<aabb>& bounds)
    {
        if(!data_ || bounds.size()!=data_->primitive_count)
            throw std::invalid_argument("BVH primitive count mismatch");
        // Validate before touching GPU-visible storage.
        for(const auto& box:bounds)
            for(int axis=0;axis<3;++axis)
            {
                float lo=(&box.minimum.x)[axis],hi=(&box.maximum.x)[axis];
                if(!finite(lo) || !finite(hi) || lo>hi)
                    throw std::invalid_argument("Invalid BVH AABB");
            }
        // The BVH's global world bounds affect only Morton key construction.
        // A refit does not re-key; large motion may justify record_build().
        vk::buffer::write(data_->primitives,bounds.data(),
                          bounds.size()*sizeof(aabb));
    }

    void tree::record_refit(vk::CommandBuffer command)
    {
        if(!data_ || !command)
            throw std::logic_error("BVH refit called before initialization");
        auto push=args(data_);
        // Preserve existing parent/child links; only clear arrival counters.
        push.sort[3]=1u;
        dispatch(data_,command,1,data_->primitive_count*2u-1u,push);
        dispatch(data_,command,3,data_->primitive_count,push);
        dispatch(data_,command,4,data_->primitive_count,push);
    }

    vk::DescriptorBufferInfo tree::primitives_descriptor()
    {
        if(!data_)throw std::logic_error("BVH not initialized");
        return vk::buffer::descriptor(data_->primitives);
    }

    vk::DescriptorBufferInfo tree::triangles_descriptor()
    {
        if(!data_ || !data_->triangle_scene)
            throw std::logic_error("Triangle BVH not initialized");
        return vk::buffer::descriptor(data_->triangles);
    }

    void tree::record_overlaps(vk::CommandBuffer command,std::uint32_t count)
    {
        if(!data_ || count>data_->max_queries)
            throw std::invalid_argument("BVH overlap query capacity exceeded");
        auto push=args(data_);
        push.sizes[2]=count;
        dispatch(data_,command,5,count,push);
    }

    void tree::record_rays(vk::CommandBuffer command,std::uint32_t count)
    {
        if(!data_ || count>data_->max_queries)
            throw std::invalid_argument("BVH ray query capacity exceeded");
        auto push=args(data_);
        push.sizes[2]=count;
        dispatch(data_,command,6,count,push);
    }

    void tree::record_triangle_rays(vk::CommandBuffer command,std::uint32_t count)
    {
        if(!data_ || !data_->triangle_scene || !command || count>data_->max_queries)
            throw std::invalid_argument("Triangle BVH query not initialized or exceeds capacity");
        auto push=args(data_);
        push.sizes[2]=count;
        push.sort[2]=1u; // exact triangle narrow phase
        dispatch(data_,command,6,count,push);
    }

    void tree::record_any_triangle_rays(vk::CommandBuffer command,std::uint32_t count)
    {
        if(!data_ || !data_->triangle_scene || !command || count>data_->max_queries)
            throw std::invalid_argument("Triangle BVH query not initialized or exceeds capacity");
        auto push=args(data_);
        push.sizes[2]=count;
        push.sort[2]=1u;
        push.sort[3]=1u; // early exit when any exact triangle hit is found
        dispatch(data_,command,6,count,push);
    }

    vk::DescriptorBufferInfo tree::nodes_descriptor()
    {
        if(!data_)throw std::logic_error("BVH not initialized");
        return vk::buffer::descriptor(data_->nodes);
    }
    vk::DescriptorBufferInfo tree::queries_descriptor()
    {
        if(!data_)throw std::logic_error("BVH not initialized");
        return vk::buffer::descriptor(data_->queries);
    }
    vk::DescriptorBufferInfo tree::hits_descriptor()
    {
        if(!data_)throw std::logic_error("BVH not initialized");
        return vk::buffer::descriptor(data_->hit_indices);
    }
    vk::DescriptorBufferInfo tree::counts_descriptor()
    {
        if(!data_)throw std::logic_error("BVH not initialized");
        return vk::buffer::descriptor(data_->hit_counts);
    }
    vk::DescriptorBufferInfo tree::rays_descriptor()
    {
        if(!data_)throw std::logic_error("BVH not initialized");
        return vk::buffer::descriptor(data_->rays);
    }
    vk::DescriptorBufferInfo tree::ray_hits_descriptor()
    {
        if(!data_)throw std::logic_error("BVH not initialized");
        return vk::buffer::descriptor(data_->ray_hits);
    }

    void tree::shutdown()
    {
        if(!data_)return;
        const auto device=vk::device::get()->handle;
        if(device)
        {
            if(data_->pool)device.destroyDescriptorPool(data_->pool);
            // shared_program owns pipelines and shared descriptor/pipeline layouts.
            for(auto* resource:{
                &data_->primitives,&data_->keys,&data_->sort_scratch,
                &data_->sort_histogram,&data_->sort_prefix,&data_->nodes,&data_->arrivals,
                &data_->queries,&data_->hit_counts,&data_->hit_indices,
                &data_->rays,&data_->ray_hits,&data_->triangles})
                if(resource->handle)vk::buffer::destroy(*resource);
        }
        delete data_;
        data_=nullptr;
    }

    namespace
    {
        tree& default_tree()
        {
            static tree value;
            return value;
        }
    }

    state* get(){return default_tree().get();}
    void init(const std::vector<aabb>& boxes,const shader_paths& shaders,
              std::uint32_t max_queries,std::uint32_t max_hits_per_query)
    {default_tree().init(boxes,shaders,max_queries,max_hits_per_query);}
    void init_triangles(const std::vector<triangle>& geometry,const shader_paths& shaders,
                        std::uint32_t max_queries,std::uint32_t max_hits_per_query)
    {default_tree().init_triangles(geometry,shaders,max_queries,max_hits_per_query);}
    void record_build(vk::CommandBuffer command){default_tree().record_build(command);}
    void update_bounds(const std::vector<aabb>& bounds){default_tree().update_bounds(bounds);}
    void record_refit(vk::CommandBuffer command){default_tree().record_refit(command);}
    void record_overlaps(vk::CommandBuffer command,std::uint32_t count)
    {default_tree().record_overlaps(command,count);}
    void record_rays(vk::CommandBuffer command,std::uint32_t count)
    {default_tree().record_rays(command,count);}
    void record_triangle_rays(vk::CommandBuffer command,std::uint32_t count)
    {default_tree().record_triangle_rays(command,count);}
    void record_any_triangle_rays(vk::CommandBuffer command,std::uint32_t count)
    {default_tree().record_any_triangle_rays(command,count);}
    vk::DescriptorBufferInfo primitives_descriptor(){return default_tree().primitives_descriptor();}
    vk::DescriptorBufferInfo triangles_descriptor(){return default_tree().triangles_descriptor();}
    vk::DescriptorBufferInfo nodes_descriptor(){return default_tree().nodes_descriptor();}
    vk::DescriptorBufferInfo queries_descriptor(){return default_tree().queries_descriptor();}
    vk::DescriptorBufferInfo hits_descriptor(){return default_tree().hits_descriptor();}
    vk::DescriptorBufferInfo counts_descriptor(){return default_tree().counts_descriptor();}
    vk::DescriptorBufferInfo rays_descriptor(){return default_tree().rays_descriptor();}
    vk::DescriptorBufferInfo ray_hits_descriptor(){return default_tree().ray_hits_descriptor();}
    void shutdown(){default_tree().shutdown();}
}
