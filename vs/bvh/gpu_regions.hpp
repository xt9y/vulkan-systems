#pragma once
#include <vs/bvh/stackless.hpp>
#include <vs/gpu/resource.hpp>
#include <vs/scene/scene.hpp>
#include <vs/spatial/world.hpp>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <unordered_map>
#include <vector>

namespace vs::bvh {
// Incremental static shadow pages. All CPU builds and region revision checks
// are independent of Vulkan. Upload(frame) only modifies the acquired frame's
// dedicated GPU buffers, never data read by another in-flight frame.
class gpu_regions {
public:
    using vertex=std::array<float,4>;
    struct alignas(16) record {
        std::array<float,4> minimum{},maximum{};
        std::array<std::uint32_t,4> span{}; // vertices, nodes, node_count, vertex_count
    };
    static_assert(sizeof(record)==48);
    struct bindings {
        vk::DescriptorBufferInfo records{},vertices{},nodes{};
    };

    void create(std::uint32_t frames,std::size_t max_regions=64,
                std::size_t max_vertices=262144,
                std::size_t max_nodes=131072);
    bool replace(spatial::region_id id,std::uint64_t revision,
                 std::vector<vertex> triangles);
    bool replace(spatial::region_id id,std::uint64_t revision,
                 std::span<const scene::triangle> triangles);
    bool erase(spatial::region_id id,std::uint64_t revision);
    // Must follow a successful frame/image fence wait. Stable buffer
    // allocations mean consumers' descriptors never change on replacement.
    bool upload(std::uint32_t image);
    bindings descriptors(std::uint32_t image) const;
    void reset() noexcept;
    std::size_t size() const noexcept {return entries_.size();}
    std::uint64_t revision() const noexcept{return serial_;}
private:
    struct entry {
        std::uint64_t revision=0;
        std::vector<vertex> vertices;
        std::vector<stackless_node> nodes;
    };
    struct location {
        std::uint64_t revision=0;
        std::uint32_t vertex_offset=0,node_offset=0;
        bool operator==(const location&) const = default;
    };
    struct frame {
        ::vs::gpu::owned_buffer records,vertices,nodes;
        std::unordered_map<spatial::region_id,location,
                           spatial::region_hash> resident;
        std::uint64_t serial=UINT64_MAX;
    };
    std::unordered_map<spatial::region_id,entry,spatial::region_hash> entries_;
    std::unordered_map<spatial::region_id,std::uint64_t,
                       spatial::region_hash> versions_;
    std::vector<std::unique_ptr<frame>> frames_;
    std::size_t region_capacity_=0,vertex_capacity_=0,node_capacity_=0;
    std::uint64_t serial_=0;
};
} // namespace vs::bvh
