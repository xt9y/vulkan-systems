#pragma once
#include "bvh.hpp"
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace bvh
{
    // Input geometry is mesh-local. One BLAS per distinct mesh; TLAS leaves are
    // world-space instance AABBs. Callers supply inverse instance transforms.
    struct instance
    {
        aabb world_bounds{};
        std::array<float,16> world_to_local{
            1.f,0.f,0.f,0.f,0.f,1.f,0.f,0.f,
            0.f,0.f,1.f,0.f,0.f,0.f,0.f,1.f
        }; // column-major 4x4
        std::uint32_t mesh=0;
    };

    struct alignas(16) instance_record
    {
        std::array<float,16> world_to_local{};
        std::array<std::uint32_t,4> metadata{}; // mesh index, reserved...
    };
    static_assert(sizeof(instance_record)==80);

    struct alignas(16) instance_hit
    {
        std::uint32_t instance=UINT32_MAX;
        std::uint32_t primitive=UINT32_MAX;
        float distance=0.f;
        std::uint32_t visited=0;
        std::uint32_t flags=0; // bit 0: TLAS/BLAS traversal stack overflow
        std::uint32_t padding[3]{};
    };
    static_assert(sizeof(instance_hit)==32);

    // BLAS hierarchy nodes and triangles are packed once into GPU arrays.
    // TLAS structures are independently rebuilt/refitted and copied into the
    // packed top-level buffer. No per-ray CPU readback or mesh draw coupling.
    class hierarchy
    {
    public:
        hierarchy()=default;
        ~hierarchy() noexcept {try{shutdown();}catch(...){}}
        hierarchy(const hierarchy&)=delete;
        hierarchy& operator=(const hierarchy&)=delete;

        void init(const std::vector<std::vector<triangle>>& meshes,
                  const std::vector<instance>& instances,
                  const shader_paths& build_shaders,
                  const std::string& traversal_shader,
                  std::uint32_t max_queries);
        void record_build(vk::CommandBuffer command);
        void update_instances(const std::vector<instance>& instances);
        // Also recopies potentially GPU-updated BLAS triangle/node arrays.
        void record_refit(vk::CommandBuffer command);
        void record_rays(vk::CommandBuffer command,std::uint32_t count,
                         bool any_hit=false);

        vk::DescriptorBufferInfo rays_descriptor() const;
        vk::DescriptorBufferInfo hits_descriptor() const;
        tree& mesh_tree(std::size_t index);
        tree& top_tree(std::size_t index);
        std::size_t mesh_count() const noexcept {return meshes_.size();}
        std::size_t instance_count() const noexcept {return instance_count_;}
        void shutdown();
    private:
        struct offsets
        {
            std::uint32_t nodes=0,base=0,unused0=0,unused1=0;
        };
        static_assert(sizeof(offsets)==16);
        struct segment
        {
            std::unique_ptr<tree> structure;
            std::uint32_t node_offset=0,base=0,count=0;
        };
        std::vector<segment> meshes_,tops_;
        vk::buffer::resource tlas_nodes_{},blas_nodes_{},triangles_{};
        vk::buffer::resource instances_{},roots_{},mesh_offsets_{};
        vk::buffer::resource rays_{},hits_{};
        vk::DescriptorSetLayout layout_{};
        vk::DescriptorPool pool_{};
        vk::DescriptorSet descriptor_{};
        vk::PipelineLayout pipeline_layout_{};
        vk::Pipeline pipeline_{};
        std::uint32_t instance_count_=0,max_queries_=0;
        void copy_nodes(vk::CommandBuffer command) const;
    };
}
