#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace vs::render {
// Asset-independent cluster hierarchy. All levels refer to the original
// vertex stream, so UVs, materials and skin weights remain consistent.
struct virtual_cluster {
    std::uint32_t first=0, count=0, parent=UINT32_MAX;
    std::array<float,3> center{};
    float radius=0, error=0;
};
struct virtual_level {
    std::vector<std::uint32_t> indices;
    std::vector<virtual_cluster> clusters;
    float error=0;
};
struct virtual_mesh {
    std::vector<virtual_level> levels;
};
// Vertices may have any stride; position_offset points at three floats.
// Builds simplified index LODs and 64-vertex/124-triangle meshlet clusters.
// Call once when loading a mesh, never from the render loop.
// Optional per-vertex float attributes/weights allow preserving UVs, normals
// and skinning across distant LODs. They are generic, not scene/model types.
virtual_mesh virtualize(const void* vertices,std::size_t vertex_count,
                        std::size_t stride,std::size_t position_offset,
                        std::span<const std::uint32_t> indices,
                        std::uint32_t max_levels=3,
                        std::span<const float> attributes={},
                        std::span<const float> attribute_weights={});
}
