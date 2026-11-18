#pragma once
#include <vs/pcloud/pcloud.hpp>
#include <vs/scene/region_runtime.hpp>
#include <vs/render/invalidation.hpp>
#include <vs/spatial/world.hpp>
#include <cstddef>
#include <span>
#include <vector>

namespace vs::pcloud {
// Consumes authoritative region scene fragments, not generated terrain.
// The caller chooses the render-coordinate origin. All fragment surfaces
// and instance transforms are interpreted in their region's 256m local frame.
// Geometry updates change only one region's CPU BVH; the GPU upload occurs
// later on the acquired swapchain image.
struct region_update_stats {std::size_t replaced=0,removed=0;};
region_update_stats synchronize_regions(
    const ::vs::scene::region_runtime& runtime,
    std::span<const ::vs::render::invalidation> changes,
    const std::vector<::vs::scene::mesh>& shared_meshes,
    ::vs::spatial::world_position render_origin={});
// Called when rebasing the entire camera-relative renderer origin, or after
// swapchain recreation. Rebuilds region BVHs from retained fragments.
// No terrain/atmosphere provider is required.
std::size_t rebuild_regions(
    const ::vs::scene::region_runtime& runtime,
    const std::vector<::vs::scene::mesh>& shared_meshes,
    ::vs::spatial::world_position render_origin={});
} // namespace vs::pcloud
