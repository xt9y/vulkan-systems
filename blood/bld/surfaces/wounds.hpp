#pragma once
#include "../core/types.hpp"
#include <unordered_map>

namespace bld::wounds
{
    // Four packed coverage bits and four packed wetness bits per UV texel.
    struct atlas
    {
        instance_id target = 0;
        std::uint32_t resolution = 128;
        std::vector<std::uint32_t> pixels;
        std::uint64_t generation = 0;
        float elapsed = 0;
    };

    struct source
    {
        instance_id target = 0;
        vec3 local_position{};
        float age = 0, next_drip = 1;
        float intensity = 1;
        std::uint32_t seed = 1;
    };

    struct state
    {
        std::uint32_t resolution = 128;
        std::uint32_t max_instances = 512;
        std::uint32_t max_marks = 256;
        std::uint32_t marks = 0;
        std::unordered_map<instance_id, atlas> atlases;
        std::vector<source> sources;
    };

    state *get();
    void init(const settings &options);
    void stamp(instance_id target, vec2 uv, float radius, float strength,
               std::uint32_t seed);
    const atlas *find(instance_id id);
    void source_at(instance_id target, vec3 local_position, float strength,
                   std::uint32_t seed);
    std::vector<vec3> update(float dt);
    void remove(instance_id id);
    void shutdown();
}
