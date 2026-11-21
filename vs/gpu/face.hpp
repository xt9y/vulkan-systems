#pragma once
#include <array>
#include <cstddef>

namespace vs::gpu
{
// Shared std430 seven-vec4 surface ABI used by PMap and Lumen.
// Each algorithm may interpret metric.w and selected alpha channels
// differently; their scene construction policies remain independent.
struct alignas(16) surface_face
{
    std::array<float, 4> origin{}, u{}, v{}, normal{}, albedo{}, emission{}, metric{};
};

static_assert(sizeof(surface_face) == 112);
static_assert(offsetof(surface_face, metric) == 96);
} // namespace vs::gpu
