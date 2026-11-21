#pragma once
#include <array>
#include <cstdint>
#include <span>
#include <vector>
#include <vs/bvh/stackless.hpp>
#include <vs/scene/scene.hpp>

namespace vs::lighting
{
// The GPU uses a fixed 128x128 octahedral GI map and up to eight glass crossings
// per light ray. Each uint stores RGB8 transmission and a normalized depth8;
// depth 255 means no intersection. Identical layout in transmission.glsl.
inline constexpr std::uint32_t transmission_resolution = 128;
inline constexpr std::uint32_t transmission_layers = 8;
inline constexpr std::uint32_t transmission_max_lights = 64;
inline constexpr std::uint32_t transmission_header_words = transmission_max_lights;

// Shared immutable lighting input. All scene construction stays with the host.
// The first 64 words are per-light 1-based map indices, followed by layered samples
// ordered by active-map index, y, x, layer. Zero means unaffected; empty scenes
// use only the 64-word header.
struct transmission_atlas
{
    std::vector<std::uint32_t> words;
    std::uint32_t active_lights = 0;
    std::uint32_t glass_triangles = 0;
    // The same ordered glass BVH supports exact RGB transmission along
    // sparse GI rays without rebuilding a second acceleration structure.
    std::vector<std::array<float, 4>> triangles;
    std::vector<::vs::bvh::stackless_node> nodes;
    std::vector<std::array<float, 4>> filters;
};

// Bake only at geometry/light-position changes. No GPU readback or per-frame
// recomputation. Mirrors and opaque casters are intentionally excluded.
// Lights without intersecting glass use the original unfiltered path.
transmission_atlas bake_transmission(
    std::span<const ::vs::scene::surface> surfaces,
    std::span<const ::vs::scene::point_light> lights);
// Also supports meshes/instances through the canonical VS scene conversion.
transmission_atlas bake_transmission(const ::vs::scene::data &scene);
} // namespace vs::lighting
