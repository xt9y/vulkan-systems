#pragma once
#include "../core/types.hpp"

namespace bld::simulation
{
    struct drop
    {
        vec3 position{}, previous{}, velocity{};
        float radius = .013f, age = 0, lifetime = 2;
        std::uint32_t seed = 1;
        std::uint64_t group = 0;
    };

    struct state
    {
        settings options{};
        std::vector<drop> particles;
    };

    state *get();
    void init(const settings &options);
    void emit(const impact &hit);
    void drip(vec3 position, std::uint32_t seed);
    void update(float dt, const collision_callback &world);
    void shutdown();
}
