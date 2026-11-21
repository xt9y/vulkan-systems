#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vs/pcloud/region_adapter.hpp>
#include <vs/render/invalidation.hpp>

namespace vs::pcloud
{
namespace
{
using vec3 = ::vs::scene::vec3;

std::vector<triangle> triangles_for(
    ::vs::spatial::region_id id, const ::vs::scene::region_fragment &fragment,
    const std::vector<::vs::scene::mesh> &meshes,
    const ::vs::spatial::world_position &render_origin)
{
    const auto offset = ::vs::spatial::camera_relative(
        {id, {0.0, 0.0, 0.0}}, render_origin);
    const vec3 shift{offset[0], offset[1], offset[2]};
    std::vector<triangle> triangles;
    triangles.reserve(fragment.surfaces.size() * 2u);

    for (const auto &surface : fragment.surfaces)
    {
        if (!surface.shadow_caster || surface.optical_type == 1u) continue;

        const vec3 a = surface.origin + shift;
        const vec3 b = a + surface.u;
        const vec3 c = a + surface.v;
        triangles.push_back({a, b, c});

        if (!surface.triangle)
            triangles.push_back({b, b + surface.v, c});
    }
    ::vs::scene::each_mesh_instance(meshes, fragment.instances,
                                    [&](const ::vs::scene::mesh &asset,
                                        const ::vs::scene::instance &instance)
                                    {
                                        if (asset.skin_id || asset.surface.alpha_mode != 0u ||
                                            instance.pattern == 2u) return;
                                        const auto &pose = instance.pose;
                                        for (std::size_t i = 0; i + 2u < asset.indices.size(); i += 3u)
                                        {
                                            const auto a = asset.indices[i], b = asset.indices[i + 1u],
                                                       c = asset.indices[i + 2u];
                                            if (a >= asset.vertices.size() || b >= asset.vertices.size() ||
                                                c >= asset.vertices.size())
                                                throw std::out_of_range("Invalid region shadow mesh indices");
                                            triangles.push_back({::vs::scene::transform_position(pose, asset.vertices[a].position) + shift, ::vs::scene::transform_position(pose, asset.vertices[b].position) + shift, ::vs::scene::transform_position(pose, asset.vertices[c].position) + shift});
                                        }
                                    });
    return triangles;
}
} // namespace
region_update_stats synchronize_regions(
    const ::vs::scene::region_runtime &runtime,
    std::span<const ::vs::render::invalidation> changes,
    const std::vector<::vs::scene::mesh> &meshes,
    ::vs::spatial::world_position origin)
{
    region_update_stats stats{};

    for (const auto &change : changes)
    {
        if (!::vs::render::any(change.changed,
                               ::vs::render::region_change::shadows) &&
            !::vs::render::any(change.changed,
                               ::vs::render::region_change::geometry) &&
            !::vs::render::any(change.changed,
                               ::vs::render::region_change::transforms) &&
            !::vs::render::any(change.changed,
                               ::vs::render::region_change::materials))
            continue;

        const auto *item = runtime.regions().get(change.id);

        if (!item) {
            forget_shadow_region(change.id);
            ++stats.removed;
        }
        else if (!item->value) {
            remove_shadow_region(change.id, item->revision);
            ++stats.removed;
        }
        else {
            auto faces = triangles_for(change.id, *item->value, meshes, origin);

            if (replace_shadow_region(change.id, item->revision, faces))
                ++stats.replaced;
        }
    }

    return stats;
}

std::size_t rebuild_regions(
    const ::vs::scene::region_runtime &runtime,
    const std::vector<::vs::scene::mesh> &meshes,
    ::vs::spatial::world_position origin)
{
    clear_shadow_regions();
    std::size_t rebuilt = 0;
    runtime.regions().each([&](::vs::spatial::region_id id,
                               std::uint64_t revision,
                               const ::vs::scene::region_fragment &region)
                           {
        auto faces=triangles_for(id,region,meshes,origin);
        if(replace_shadow_region(id,revision,faces))++rebuilt; });
    return rebuilt;
}
} // namespace vs::pcloud
