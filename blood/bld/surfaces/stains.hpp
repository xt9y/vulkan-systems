#pragma once
#include "../core/types.hpp"
#include <fld/surface/film.hpp>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace bld::surface
{
    struct mark
    {
        vec2 uv{};
        vec2 radius_uv{.015f,.015f};
        float age=0, volume_ml=.1f;
        float angle=0, wetness=1;
        std::uint32_t seed=1;
    };

    struct atlas
    {
        instance_id target=0;
        std::uint32_t resolution=256;
        std::vector<std::uint32_t> pixels;
        // Flatten evicted marks here so stains survive the active-mark budget.
        // Allocated lazily, only when the first mark is evicted.
        std::vector<std::uint32_t> baked_pixels;
        std::vector<mark> marks; // impact history, not a procedural mask
        fld::surface::grid film;
        std::uint64_t generation=0;
        float update_accumulator=0;
    };

    struct rigid_stain
    {
        instance_id target=0;
        mat4 local=mat4::identity();
        mat4 world=mat4::identity();
        vec3 normal{0,1,0};
        float radius=.04f, age=0, wetness=1, volume_ml=.1f;
        std::uint32_t seed=1;
        substrate kind=substrate::painted;
        fld::surface::grid film;
        std::vector<std::uint32_t> pixels;
    };

    struct target
    {
        substrate kind=substrate::painted;
        mat4 transform=mat4::identity();
        bool skinned=false;
        // World-space distance covered by one UV unit on this mesh.
        // Supply the local hit-triangle metric to prevent oversized marks.
        vec2 metres_per_uv{1.f,1.f};
    };

    struct state
    {
        options settings{};
        std::unordered_map<instance_id,target> targets;
        std::unordered_map<instance_id,atlas> atlases;
        std::vector<rigid_stain> stains;
        std::unordered_set<instance_id> rigid_targets;
        std::uint64_t generation=1;
        float rigid_update_accumulator=0.f;
    };

    state *get();
    void init(const options &settings);
    void register_target(instance_id id, substrate kind, mat4 transform,
                         bool skinned=false,
                         vec2 metres_per_uv={1.f,1.f});
    void set_transform(instance_id id, mat4 transform);
    // Optional geometry-aware chart for curvature and local tangent flow.
    void register_geometry(instance_id id,
        const std::vector<fld::surface::chart_vertex>& vertices,
        const std::vector<std::uint32_t>& indices);
    void remove_target(instance_id id);
    void deposit(instance_id id, vec3 position, vec3 normal, vec2 uv,
                 bool uv_valid, float volume_ml, float velocity,
                 std::uint32_t seed);
    void update(float dt);
    const atlas *find(instance_id id);
    // Cubic metres in the surface/porous reservoirs (separate from 3D drops).
    double volume();
    void clear();
    void shutdown();
}
