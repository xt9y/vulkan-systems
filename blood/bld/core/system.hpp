#pragma once
#include "types.hpp"
#include "../simulation/solver.hpp"
#include "../surfaces/decals.hpp"
#include "../surfaces/wounds.hpp"

namespace bld::system
{
    struct state
    {
        settings options{};
        std::uint64_t sequence = 1;
        double accumulator = 0;
        std::vector<impact> pending_backdrops;
        std::uint64_t steps = 0;
    };

    state *get();
    void init(const settings &options = {});
    impact hit(impact event);
    void apply(const impact &event);
    void update(float dt, const collision_callback &world = {});
    void clear();
    void shutdown();
}
