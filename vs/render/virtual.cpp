#include <algorithm>
#include <cmath>
#include <limits>
#include <meshoptimizer.h>
#include <stdexcept>
#include <utility>
#include <vs/render/virtual.hpp>

namespace vs::render
{
namespace
{
void clusterize(virtual_level &level, const float *positions,
                std::size_t vertex_count, std::size_t stride)
{
    constexpr std::size_t max_vertices = 64, max_triangles = 124;

    if (level.indices.empty()) return;

    const auto bound = meshopt_buildMeshletsBound(
        level.indices.size(), max_vertices, max_triangles);
    std::vector<meshopt_Meshlet> groups(bound);
    std::vector<unsigned int> vertices(bound * max_vertices);
    std::vector<unsigned char> triangles(bound * max_triangles * 3);

    const auto n = meshopt_buildMeshlets(
        groups.data(), vertices.data(), triangles.data(),
        level.indices.data(), level.indices.size(), positions,
        vertex_count, stride, max_vertices, max_triangles, 0.0f);
    std::vector<std::uint32_t> ordered;
    ordered.reserve(level.indices.size());
    level.clusters.reserve(n);

    for (std::size_t i = 0; i < n; ++i)
    {
        const auto &g = groups[i];

        if (!g.triangle_count) continue;

        const auto bounds = meshopt_computeMeshletBounds(
            vertices.data() + g.vertex_offset, triangles.data() + g.triangle_offset,
            g.triangle_count, positions, vertex_count, stride);
        virtual_cluster c{};
        c.first = std::uint32_t(ordered.size());
        c.count = g.triangle_count * 3u;
        c.center = {bounds.center[0], bounds.center[1], bounds.center[2]};
        c.radius = bounds.radius;
        c.error = level.error;

        for (std::size_t k = 0; k < std::size_t(c.count); ++k)
            ordered.push_back(vertices[g.vertex_offset +
                                       triangles[g.triangle_offset + k]]);
        level.clusters.push_back(c);
    }

    level.indices = std::move(ordered);
}
} // namespace
virtual_mesh virtualize(const void *vertices, std::size_t vertex_count,
                        std::size_t stride, std::size_t position_offset,
                        std::span<const std::uint32_t> indices,
                        std::uint32_t max_levels,
                        std::span<const float> attributes,
                        std::span<const float> attribute_weights)
{
    virtual_mesh result;

    if (!vertex_count || indices.empty()) return result;
    if (!vertices || stride < position_offset + sizeof(float) * 3u ||
        indices.size() % 3u || max_levels == 0u)
        throw std::invalid_argument("Invalid virtual geometry source");

    for (const auto id : indices)
        if (id >= vertex_count)
            throw std::out_of_range("Virtual geometry has invalid indices");

    const auto attribute_count = attribute_weights.size();

    if (attribute_count > 32u ||
        attributes.empty() != attribute_weights.empty() ||
        (attribute_count && (attributes.size() % attribute_count != 0u ||
                             attributes.size() / attribute_count != vertex_count)))
        throw std::invalid_argument("Invalid virtual geometry attributes");

    const auto *positions = reinterpret_cast<const float *>(
        static_cast<const unsigned char *>(vertices) + position_offset);
    virtual_level finest{};
    finest.indices.assign(indices.begin(), indices.end());
    result.levels.push_back(std::move(finest));

    for (std::uint32_t level = 1; level < max_levels; ++level)
    {
        const auto &source = result.levels.back();

        if (source.indices.size() < 192u) break;

        const auto target = std::max<std::size_t>(
            96u, ((source.indices.size() / 2u) / 3u) * 3u);
        if (target >= source.indices.size()) break;
        virtual_level coarse{};
        coarse.indices.resize(source.indices.size());

        float error = 0.f;
        const auto n = attribute_count ? meshopt_simplifyWithAttributes(
                                             coarse.indices.data(), source.indices.data(), source.indices.size(),
                                             positions, vertex_count, stride, attributes.data(),
                                             attribute_count * sizeof(float), attribute_weights.data(),
                                             attribute_count, nullptr, target, 1.0f,
                                             meshopt_SimplifyLockBorder, &error)
                                       : meshopt_simplify(
                                             coarse.indices.data(), source.indices.data(), source.indices.size(),
                                             positions, vertex_count, stride, target, 1.0f, 0u, &error);
        if (n >= source.indices.size() * 9u / 10u || n < 3u) break;
        coarse.indices.resize(n);
        coarse.error = std::max(source.error, error);
        result.levels.push_back(std::move(coarse));
    }

    for (auto &level : result.levels)
        clusterize(level, positions, vertex_count, stride);
    // A cluster in a fine level belongs to its nearest coarse aggregate.
    // Parent links allow GPU traversal/streaming without requiring a world.
    for (std::size_t level = 0; level + 1 < result.levels.size(); ++level)
    {
        auto &parents = result.levels[level + 1].clusters;

        if (parents.empty()) continue;

        for (auto &child : result.levels[level].clusters)
        {
            float best = std::numeric_limits<float>::max();

            for (std::size_t p = 0; p < parents.size(); ++p)
            {
                const auto &parent = parents[p];

                const float x = child.center[0] - parent.center[0];
                const float y = child.center[1] - parent.center[1];
                const float z = child.center[2] - parent.center[2];
                const float d = x * x + y * y + z * z;

                if (d < best) {
                    best = d;
                    child.parent = std::uint32_t(p);
                }
            }

            if (child.parent != UINT32_MAX) {
                // A parent must conservatively contain every assigned child.
                // This remains valid for hierarchy traversal and residency
                // even when simplification changes triangle connectivity.
                auto &parent = parents[child.parent];
                parent.radius = std::max(parent.radius,
                                         std::sqrt(best) + child.radius);
            }
        }
    }

    return result;
}
} // namespace vs::render
