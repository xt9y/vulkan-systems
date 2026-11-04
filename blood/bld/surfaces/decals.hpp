#pragma once
#include "../core/types.hpp"
#include <unordered_map>

namespace bld::surfaces
{
    struct decal
    {
        instance_id target = 0;
        mat4 world = mat4::identity();
        mat4 local = mat4::identity();
        float age = 0, lifetime = 240;
        float radius = .2f, final_radius = .2f;
        float wetness = 1;
        std::uint32_t variant = 0;
        bool pool = false;
    };

    struct projector
    {
        mat4 world_to_local = mat4::identity();
        vec3 normal{0, 1, 0};
        float wetness = 1;
        std::uint32_t variant = 0;
        bool pool = false;
    };

    struct state
    {
        std::uint32_t capacity = 1024, max_pools = 64;
        std::vector<decal> decals;
        std::unordered_map<instance_id, mat4> transforms;
    };

    state *get();
    void init(const settings &options);
    void set_transform(instance_id id, const mat4 &model);
    void remove_transform(instance_id id);
    void add(const collision &hit, float size, std::uint32_t seed,
             bool pool = false, float stretch = 1.0f);
    void add(instance_id target, vec3 position, vec3 normal, float size,
             std::uint32_t seed, bool pool = false, float stretch = 1.0f);
    void update(float dt);
    std::vector<projector> resolve();
    void shutdown();
}
