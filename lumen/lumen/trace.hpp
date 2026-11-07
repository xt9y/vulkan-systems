#pragma once
#include <bvh/bvh.hpp>
#include <array>
#include <vector>

namespace lumen::trace {

// CPU construction of a static, median-split BVH; GPU traversal uses the
// VulkanBVH std430 node and triangle layouts. No ray CPU readback.
// The GPU-built BVH/TLAS backend can replace this builder without changing
// scene shading or descriptor layouts.
struct mesh {
    std::vector<bvh::node> nodes;
    std::vector<bvh::triangle> triangles;
};

mesh build(const std::vector<std::array<float,3>>& triangle_vertices);

} // namespace lumen::trace
