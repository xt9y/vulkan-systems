#include "hierarchy.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace bvh
{
    namespace
    {
        constexpr std::uint32_t chunk_size=65536u;

        vk::buffer::resource allocate(std::size_t bytes,
                                      vk::BufferUsageFlags extra={},
                                      bool host=false)
        {
            if(!bytes)throw std::invalid_argument("Zero-sized hierarchy buffer");
            return vk::buffer::create(bytes,
                vk::BufferUsageFlagBits::eStorageBuffer|extra,
                host?(vk::MemoryPropertyFlagBits::eHostVisible |
                      vk::MemoryPropertyFlagBits::eHostCoherent):
                     vk::MemoryPropertyFlags(vk::MemoryPropertyFlagBits::eDeviceLocal));
        }

        void memory_barrier(vk::CommandBuffer cmd,
                            vk::PipelineStageFlags src_stage,
                            vk::PipelineStageFlags dst_stage,
                            vk::AccessFlags src_access,
                            vk::AccessFlags dst_access)
        {
            vk::MemoryBarrier barrier{};
            barrier.srcAccessMask=src_access;
            barrier.dstAccessMask=dst_access;
            cmd.pipelineBarrier(src_stage,dst_stage,{}, {barrier},{},{});
        }

        std::uint32_t checked_offset(std::uint64_t n)
        {
            if(n>std::numeric_limits<std::uint32_t>::max())
                throw std::overflow_error("BVH address range exceeded");
            return static_cast<std::uint32_t>(n);
        }

        bool finite_bounds(const aabb& box)
        {
            for(int i=0;i<3;++i)
            {
                const float lo=(&box.minimum.x)[i];
                const float hi=(&box.maximum.x)[i];
                if(!std::isfinite(lo) || !std::isfinite(hi) || lo>hi)
                    return false;
            }
            return true;
        }

        void validate_instance(const instance& object,std::size_t mesh_count)
        {
            if(object.mesh>=mesh_count || !finite_bounds(object.world_bounds))
                throw std::invalid_argument("Invalid BVH mesh instance");
            for(float v:object.world_to_local)
                if(!std::isfinite(v))
                    throw std::invalid_argument("Non-finite inverse instance transform");
        }
    }

    void hierarchy::init(const std::vector<std::vector<triangle>>& geometry,
                         const std::vector<instance>& instances,
                         const shader_paths& build_shaders,
                         const std::string& traversal_shader,
                         std::uint32_t max_queries)
    {
        if(!meshes_.empty() || !tops_.empty() || pipeline_ ||
           geometry.empty() || instances.empty() || !max_queries ||
           max_queries>65536u || traversal_shader.empty())
            throw std::invalid_argument("Invalid or already initialized BVH hierarchy");
        for(const auto& g:geometry)
            if(g.empty() || g.size()>65536u)
                throw std::invalid_argument("Invalid mesh BLAS size");
        for(const auto& object:instances)
            validate_instance(object,geometry.size());

        const auto device=vk::device::get()->handle;
        if(!device)throw std::logic_error("Vulkan device is not initialized");
        instance_count_=checked_offset(instances.size());
        max_queries_=max_queries;

        try
        {
            std::vector<triangle> packed_triangles;
            std::vector<offsets> mesh_info;
            std::uint64_t mesh_node_count=0;
            for(const auto& triangles:geometry)
            {
                segment mesh{};
                mesh.structure=std::make_unique<tree>();
                mesh.structure->init_triangles(triangles,build_shaders);
                mesh.node_offset=checked_offset(mesh_node_count);
                mesh.base=checked_offset(packed_triangles.size());
                mesh.count=static_cast<std::uint32_t>(triangles.size());
                mesh_node_count+=triangles.size()*2u-1u;
                checked_offset(mesh_node_count);
                mesh_info.push_back({mesh.node_offset,mesh.base,0,0});
                packed_triangles.insert(packed_triangles.end(),
                                        triangles.begin(),triangles.end());
                meshes_.push_back(std::move(mesh));
            }

            std::vector<offsets> root_info;
            std::uint64_t top_node_count=0;
            for(std::size_t begin=0;begin<instances.size();begin+=chunk_size)
            {
                const auto end=std::min(instances.size(),begin+chunk_size);
                std::vector<aabb> bounds;
                bounds.reserve(end-begin);
                for(std::size_t i=begin;i<end;++i)
                    bounds.push_back(instances[i].world_bounds);
                segment top{};
                top.structure=std::make_unique<tree>();
                top.structure->init(bounds,build_shaders);
                top.node_offset=checked_offset(top_node_count);
                top.base=checked_offset(begin);
                top.count=checked_offset(end-begin);
                top_node_count+=2u*(end-begin)-1u;
                checked_offset(top_node_count);
                root_info.push_back({top.node_offset,top.base,0,0});
                tops_.push_back(std::move(top));
            }

            tlas_nodes_=allocate(top_node_count*sizeof(node),
                                 vk::BufferUsageFlagBits::eTransferDst);
            blas_nodes_=allocate(mesh_node_count*sizeof(node),
                                 vk::BufferUsageFlagBits::eTransferDst);
            triangles_=allocate(packed_triangles.size()*sizeof(triangle),
                                vk::BufferUsageFlagBits::eTransferDst,true);
            vk::buffer::write(triangles_,packed_triangles.data(),
                              packed_triangles.size()*sizeof(triangle));
            instances_=allocate(instances.size()*sizeof(instance_record),{},true);
            roots_=allocate(root_info.size()*sizeof(offsets),{},true);
            mesh_offsets_=allocate(mesh_info.size()*sizeof(offsets),{},true);
            rays_=allocate(std::size_t(max_queries)*sizeof(ray),{},true);
            hits_=allocate(std::size_t(max_queries)*sizeof(instance_hit));
            vk::buffer::write(roots_,root_info.data(),root_info.size()*sizeof(offsets));
            vk::buffer::write(mesh_offsets_,mesh_info.data(),
                              mesh_info.size()*sizeof(offsets));

            std::array<vk::DescriptorSetLayoutBinding,8> bindings{};
            for(std::uint32_t i=0;i<bindings.size();++i)
            {
                auto& b=bindings[i];
                b.binding=i;
                b.descriptorCount=1;
                b.descriptorType=vk::DescriptorType::eStorageBuffer;
                b.stageFlags=vk::ShaderStageFlagBits::eCompute;
            }
            vk::DescriptorSetLayoutCreateInfo info{};
            info.bindingCount=bindings.size();
            info.pBindings=bindings.data();
            layout_=vk::descriptor_layout::create(info);

            vk::PushConstantRange range{};
            range.size=sizeof(std::uint32_t)*4u;
            range.stageFlags=vk::ShaderStageFlagBits::eCompute;
            vk::PipelineLayoutCreateInfo layout{};
            layout.setLayoutCount=1;
            layout.pSetLayouts=&layout_;
            layout.pushConstantRangeCount=1;
            layout.pPushConstantRanges=&range;
            pipeline_layout_=vk::pipeline_layout::create(layout);

            auto shader=vk::shader::create(vk::shader::read(traversal_shader));
            try
            {
                vk::ComputePipelineCreateInfo pipeline{};
                pipeline.stage.stage=vk::ShaderStageFlagBits::eCompute;
                pipeline.stage.module=shader;
                pipeline.stage.pName="main";
                pipeline.layout=pipeline_layout_;
                pipeline_=vk::pipeline::compute(pipeline);
            }
            catch(...)
            {
                vk::shader::destroy(shader);
                throw;
            }
            vk::shader::destroy(shader);

            vk::DescriptorPoolSize pool_size{vk::DescriptorType::eStorageBuffer,8};
            vk::DescriptorPoolCreateInfo pool{};
            pool.maxSets=1;
            pool.poolSizeCount=1;
            pool.pPoolSizes=&pool_size;
            pool_=device.createDescriptorPool(pool);
            vk::DescriptorSetAllocateInfo allocation{};
            allocation.descriptorPool=pool_;
            allocation.descriptorSetCount=1;
            allocation.pSetLayouts=&layout_;
            descriptor_=device.allocateDescriptorSets(allocation).front();

            std::array<vk::DescriptorBufferInfo,8> descriptors{{
                vk::buffer::descriptor(tlas_nodes_),
                vk::buffer::descriptor(blas_nodes_),
                vk::buffer::descriptor(triangles_),
                vk::buffer::descriptor(instances_),
                vk::buffer::descriptor(rays_),
                vk::buffer::descriptor(hits_),
                vk::buffer::descriptor(roots_),
                vk::buffer::descriptor(mesh_offsets_)
            }};
            std::array<vk::WriteDescriptorSet,8> writes{};
            for(std::uint32_t i=0;i<writes.size();++i)
            {
                auto& w=writes[i];
                w.dstSet=descriptor_;
                w.dstBinding=i;
                w.descriptorCount=1;
                w.descriptorType=vk::DescriptorType::eStorageBuffer;
                w.pBufferInfo=&descriptors[i];
            }
            device.updateDescriptorSets(writes,{});
            update_instances(instances);
        }
        catch(...)
        {
            shutdown();
            throw;
        }
    }

    void hierarchy::copy_nodes(vk::CommandBuffer cmd) const
    {
        for(const auto& segment:tops_)
        {
            const vk::BufferCopy copy{0,
                vk::DeviceSize(segment.node_offset)*sizeof(node),
                vk::DeviceSize(segment.count*2u-1u)*sizeof(node)};
            cmd.copyBuffer(segment.structure->get()->nodes.handle,
                           tlas_nodes_.handle,{copy});
        }
        for(const auto& segment:meshes_)
        {
            const vk::BufferCopy copy{0,
                vk::DeviceSize(segment.node_offset)*sizeof(node),
                vk::DeviceSize(segment.count*2u-1u)*sizeof(node)};
            cmd.copyBuffer(segment.structure->get()->nodes.handle,
                           blas_nodes_.handle,{copy});
            const vk::BufferCopy triangles_copy{0,
                vk::DeviceSize(segment.base)*sizeof(triangle),
                vk::DeviceSize(segment.count)*sizeof(triangle)};
            cmd.copyBuffer(segment.structure->get()->triangles.handle,
                           triangles_.handle,{triangles_copy});
        }
    }

    void hierarchy::record_build(vk::CommandBuffer command)
    {
        if(!command || tops_.empty())throw std::logic_error("BVH hierarchy not ready");
        for(auto& mesh:meshes_)mesh.structure->record_build(command);
        for(auto& top:tops_)top.structure->record_build(command);
        memory_barrier(command,vk::PipelineStageFlagBits::eComputeShader,
                       vk::PipelineStageFlagBits::eTransfer,
                       vk::AccessFlagBits::eShaderWrite,
                       vk::AccessFlagBits::eTransferRead);
        copy_nodes(command);
        memory_barrier(command,vk::PipelineStageFlagBits::eTransfer,
                       vk::PipelineStageFlagBits::eComputeShader,
                       vk::AccessFlagBits::eTransferWrite,
                       vk::AccessFlagBits::eShaderRead);
    }

    void hierarchy::update_instances(const std::vector<instance>& source)
    {
        if(tops_.empty() || source.size()!=instance_count_)
            throw std::invalid_argument("Cannot change hierarchy instance count");
        for(const auto& object:source)
            validate_instance(object,meshes_.size());

        std::vector<instance_record> records;
        records.reserve(source.size());
        for(const auto& object:source)
        {
            instance_record value{};
            value.world_to_local=object.world_to_local;
            value.metadata[0]=object.mesh;
            records.push_back(value);
        }
        // Caller must wait for outstanding GPU reads before editing host memory.
        vk::buffer::write(instances_,records.data(),records.size()*sizeof(records[0]));
        for(const auto& top:tops_)
        {
            std::vector<aabb> boxes;
            boxes.reserve(top.count);
            for(std::size_t i=top.base;i<std::size_t(top.base)+top.count;++i)
                boxes.push_back(source[i].world_bounds);
            top.structure->update_bounds(boxes);
        }
    }

    void hierarchy::record_refit(vk::CommandBuffer command)
    {
        if(!command || tops_.empty())throw std::logic_error("BVH hierarchy not ready");
        for(auto& mesh:meshes_)mesh.structure->record_refit(command);
        for(auto& top:tops_)top.structure->record_refit(command);
        memory_barrier(command,vk::PipelineStageFlagBits::eComputeShader,
                       vk::PipelineStageFlagBits::eTransfer,
                       vk::AccessFlagBits::eShaderWrite,
                       vk::AccessFlagBits::eTransferRead);
        copy_nodes(command);
        memory_barrier(command,vk::PipelineStageFlagBits::eTransfer,
                       vk::PipelineStageFlagBits::eComputeShader,
                       vk::AccessFlagBits::eTransferWrite,
                       vk::AccessFlagBits::eShaderRead);
    }

    void hierarchy::record_rays(vk::CommandBuffer command,std::uint32_t count,
                                bool any_hit)
    {
        if(!command || !pipeline_ || count>max_queries_)
            throw std::invalid_argument("BVH hierarchy ray query exceeds capacity");
        if(!count)return;
        const std::array<std::uint32_t,4> push{
            count,checked_offset(tops_.size()),any_hit?1u:0u,0u
        };
        vk::record::pipeline(command,vk::PipelineBindPoint::eCompute,pipeline_);
        command.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                   pipeline_layout_,0,{descriptor_},{});
        command.pushConstants(pipeline_layout_,vk::ShaderStageFlagBits::eCompute,
                              0,sizeof(push),push.data());
        vk::record::dispatch(command,(count+63u)/64u,1u,1u);
        memory_barrier(command,vk::PipelineStageFlagBits::eComputeShader,
                       vk::PipelineStageFlagBits::eComputeShader,
                       vk::AccessFlagBits::eShaderWrite,
                       vk::AccessFlagBits::eShaderRead);
    }

    vk::DescriptorBufferInfo hierarchy::rays_descriptor() const
    {
        if(!rays_.handle)throw std::logic_error("BVH hierarchy not initialized");
        return vk::buffer::descriptor(rays_);
    }
    vk::DescriptorBufferInfo hierarchy::hits_descriptor() const
    {
        if(!hits_.handle)throw std::logic_error("BVH hierarchy not initialized");
        return vk::buffer::descriptor(hits_);
    }
    tree& hierarchy::mesh_tree(std::size_t index)
    {
        return *meshes_.at(index).structure;
    }
    tree& hierarchy::top_tree(std::size_t index)
    {
        return *tops_.at(index).structure;
    }

    void hierarchy::shutdown()
    {
        const auto device=vk::device::get()->handle;
        if(device)
        {
            if(pipeline_)vk::pipeline::destroy(pipeline_);
            if(pipeline_layout_)vk::pipeline_layout::destroy(pipeline_layout_);
            if(pool_)device.destroyDescriptorPool(pool_);
            if(layout_)vk::descriptor_layout::destroy(layout_);
            for(auto* buffer:{
                &tlas_nodes_,&blas_nodes_,&triangles_,&instances_,
                &roots_,&mesh_offsets_,&rays_,&hits_})
                if(buffer->handle)vk::buffer::destroy(*buffer);
        }
        pipeline_=nullptr;
        pipeline_layout_=nullptr;
        pool_=nullptr;
        layout_=nullptr;
        descriptor_=nullptr;
        meshes_.clear();
        tops_.clear();
        instance_count_=0;
        max_queries_=0;
    }
}
