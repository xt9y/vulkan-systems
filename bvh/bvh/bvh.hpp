#pragma once
#include <vk/vk.hpp>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace bvh
{
    using vec4=vk::util::math::vec4;
    // All nodes and leaves use std430-compatible, 16-byte-aligned fields.
    // The library accepts arbitrary primitive world-space axis-aligned bounds:
    // triangles, meshes, rigid bodies, decals, fluid colliders, etc.
    struct alignas(16) aabb
    {
        vec4 minimum{},maximum{};
    };
    static_assert(sizeof(aabb)==32);

    // Three world-space vertices, std430-compatible, indexed by BVH leaf ID.
    struct alignas(16) triangle
    {
        vec4 a{},b{},c{};
    };
    static_assert(sizeof(triangle)==48);

    struct alignas(16) node
    {
        vec4 minimum{},maximum{};
        std::array<std::uint32_t,4> links{}; // left,right,parent,primitive
    };
    static_assert(sizeof(node)==48);

    struct alignas(16) ray
    {
        vec4 origin_max{};  // xyz origin, w maximum distance
        vec4 direction_min{}; // xyz normalized ray, w minimum distance
    };
    struct alignas(16) ray_hit
    {
        std::uint32_t primitive=UINT32_MAX;
        float distance=0.f;
        std::uint32_t visited=0,flags=0; // bit 0: stack overflow
    };

    // Shader order: keys, clear, topology, leaves, refit, overlap,
    // ray, radix_histogram, radix_prefix, radix_scatter.
    using shader_paths=std::array<std::string,10>;
    struct shared_pipelines;
    struct state
    {
        vk::buffer::resource primitives,keys,sort_scratch,sort_histogram,sort_prefix,nodes,arrivals,triangles;
        vk::buffer::resource queries,hit_counts,hit_indices,rays,ray_hits;
        vk::DescriptorSetLayout layout{};
        vk::DescriptorPool pool{};
        vk::DescriptorSet descriptors{};
        vk::PipelineLayout pipeline_layout{};
        std::array<vk::Pipeline,10> pipelines{};
        std::uint32_t primitive_count=0,padded_count=0;
        std::uint32_t max_queries=0,max_hits_per_query=0;
        bool triangle_scene=false;
        std::shared_ptr<shared_pipelines> shared_program;
        vec4 world_min{},world_max{};
    };

    // Independent GPU acceleration structure: supports multiple BLAS/TLAS trees.
    // Destruction requires all referencing GPU submissions to have finished.
    class tree
    {
    public:
        tree()=default;
        ~tree() noexcept {try{shutdown();}catch(...){}}
        tree(const tree&)=delete;
        tree& operator=(const tree&)=delete;
        tree(tree&&)=delete;
        tree& operator=(tree&&)=delete;
        state* get() const noexcept {return data_;}
        void init(const std::vector<aabb>& bounds,const shader_paths& shaders,
                  std::uint32_t max_queries=0,std::uint32_t max_hits_per_query=16);
        void init_triangles(const std::vector<triangle>& geometry,const shader_paths& shaders,
                            std::uint32_t max_queries=0,std::uint32_t max_hits_per_query=16);
        void record_build(vk::CommandBuffer command);
        void update_bounds(const std::vector<aabb>& bounds);
        void record_refit(vk::CommandBuffer command);
        void record_overlaps(vk::CommandBuffer command,std::uint32_t count);
        void record_rays(vk::CommandBuffer command,std::uint32_t count);
        void record_triangle_rays(vk::CommandBuffer command,std::uint32_t count);
        void record_any_triangle_rays(vk::CommandBuffer command,std::uint32_t count);
        vk::DescriptorBufferInfo primitives_descriptor();
        vk::DescriptorBufferInfo triangles_descriptor();
        vk::DescriptorBufferInfo nodes_descriptor();
        vk::DescriptorBufferInfo queries_descriptor();
        vk::DescriptorBufferInfo hits_descriptor();
        vk::DescriptorBufferInfo counts_descriptor();
        vk::DescriptorBufferInfo rays_descriptor();
        vk::DescriptorBufferInfo ray_hits_descriptor();
        void shutdown();
    private:
        state* data_=nullptr;
        void init_internal(const std::vector<aabb>& bounds,const shader_paths& shaders,
                           std::uint32_t max_queries,std::uint32_t max_hits_per_query,
                           const std::vector<triangle>* geometry);
    };

    // Compatibility functions use one default tree. Prefer owning tree objects
    // when handling multiple meshes and a separate top-level instance BVH.
    state* get();
    // Initialization uploads the static bounds once. No CPU BVH build.
    void init(const std::vector<aabb>& bounds,const shader_paths& shaders,
              std::uint32_t max_queries=0,std::uint32_t max_hits_per_query=16);
    // Exact two-sided ray-triangle visibility; the BVH is built over triangle AABBs.
    // Triangle input and hit IDs use the same ordering. Geometry is static until shutdown.
    void init_triangles(const std::vector<triangle>& geometry,const shader_paths& shaders,
                        std::uint32_t max_queries=0,std::uint32_t max_hits_per_query=16);
    // Call once for immutable bounds, or after bounds are updated on GPU.
    // The caller provides command-buffer ordering and submission fences.
    void record_build(vk::CommandBuffer command);
    // Refit existing topology in O(N) after changing primitive bounds.
    // Avoids Morton re-sorting; caller ensures previous GPU work is complete.
    void update_bounds(const std::vector<aabb>& bounds);
    void record_refit(vk::CommandBuffer command);
    vk::DescriptorBufferInfo primitives_descriptor();
    // For GPU-driven skinned meshes: update the triangle and AABB storage via
    // compute passes, then record_refit after the appropriate memory barrier.
    vk::DescriptorBufferInfo triangles_descriptor();
    // Query input/output stay GPU-resident and can be read by other passes.
    void record_overlaps(vk::CommandBuffer command,std::uint32_t count);
    // record_rays remains the existing ray-AABB query for arbitrary primitives.
    void record_rays(vk::CommandBuffer command,std::uint32_t count);
    // Exact closest-hit ray-triangle queries. Only valid after init_triangles.
    void record_triangle_rays(vk::CommandBuffer command,std::uint32_t count);
    // Early-exit on the first triangle hit: useful for shadow/GI visibility rays.
    void record_any_triangle_rays(vk::CommandBuffer command,std::uint32_t count);
    vk::DescriptorBufferInfo nodes_descriptor();
    vk::DescriptorBufferInfo queries_descriptor();
    vk::DescriptorBufferInfo hits_descriptor();
    vk::DescriptorBufferInfo counts_descriptor();
    vk::DescriptorBufferInfo rays_descriptor();
    vk::DescriptorBufferInfo ray_hits_descriptor();
    void shutdown();
}
