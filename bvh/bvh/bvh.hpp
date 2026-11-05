#pragma once
#include <vk/vk.hpp>
#include <array>
#include <cstdint>
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

    // shader order: keys, compare-exchange, clear, topology,
    // leaves, refit, overlap, ray. GPU build: Morton+bitonic+radix topology.
    using shader_paths=std::array<std::string,8>;
    struct state
    {
        vk::buffer::resource primitives,keys,nodes,arrivals;
        vk::buffer::resource queries,hit_counts,hit_indices,rays,ray_hits;
        vk::DescriptorSetLayout layout{};
        vk::DescriptorPool pool{};
        vk::DescriptorSet descriptors{};
        vk::PipelineLayout pipeline_layout{};
        std::array<vk::Pipeline,8> pipelines{};
        std::uint32_t primitive_count=0,padded_count=0;
        std::uint32_t max_queries=0,max_hits_per_query=0;
        vec4 world_min{},world_max{};
    };

    state* get();
    // Initialization uploads the static bounds once. No CPU BVH build.
    void init(const std::vector<aabb>& bounds,const shader_paths& shaders,
              std::uint32_t max_queries=0,std::uint32_t max_hits_per_query=16);
    // Call once for immutable bounds, or after bounds are updated on GPU.
    // The caller provides command-buffer ordering and submission fences.
    void record_build(vk::CommandBuffer command);
    // Query input/output stay GPU-resident and can be read by other passes.
    void record_overlaps(vk::CommandBuffer command,std::uint32_t count);
    void record_rays(vk::CommandBuffer command,std::uint32_t count);
    vk::DescriptorBufferInfo nodes_descriptor();
    vk::DescriptorBufferInfo queries_descriptor();
    vk::DescriptorBufferInfo hits_descriptor();
    vk::DescriptorBufferInfo counts_descriptor();
    vk::DescriptorBufferInfo rays_descriptor();
    vk::DescriptorBufferInfo ray_hits_descriptor();
    void shutdown();
}
