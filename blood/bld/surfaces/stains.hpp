#pragma once
#include "../core/types.hpp"
#include <unordered_map>
#include <vector>

namespace bld::surface
{
    struct mark
    {
        vec2 uv{};
        float radius=.015f, age=0, volume_ml=.1f;
        float angle=0, wetness=1;
        std::uint32_t seed=1;
    };

    struct atlas
    {
        instance_id target=0;
        std::uint32_t resolution=256;
        std::vector<std::uint32_t> pixels;
        std::vector<mark> marks;
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
    };

    struct target
    {
        substrate kind=substrate::painted;
        mat4 transform=mat4::identity();
        bool skinned=false;
    };

    struct state
    {
        options settings{};
        std::unordered_map<instance_id,target> targets;
        std::unordered_map<instance_id,atlas> atlases;
        std::vector<rigid_stain> stains;
        std::uint64_t generation=1;
    };

    state *get();
    void init(const options &settings);
    void register_target(instance_id id, substrate kind, mat4 transform,
                         bool skinned=false);
    void set_transform(instance_id id, mat4 transform);
    void remove_target(instance_id id);
    void deposit(instance_id id, vec3 position, vec3 normal, vec2 uv,
                 bool uv_valid, float volume_ml, float velocity,
                 std::uint32_t seed);
    void update(float dt);
    const atlas *find(instance_id id);
    void clear();
    void shutdown();
}
