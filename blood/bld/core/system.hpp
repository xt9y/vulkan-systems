#pragma once
#include "types.hpp"

namespace bld::system
{
    struct state
    {
        options settings{};
        std::uint32_t phase=0;
        std::uint64_t sequence=1;
        bool owns_fluid=false;
        std::uint64_t impacts=0, deposited=0;
    };

    state *get();
    void init(const options &options = {});
    impact hit(impact event);
    void apply(const impact &event);
    // The caller supplies scene collision; all fluid mechanics live in fld.
    void update(float dt, const collision_callback &collision = {});
    void clear();
    void shutdown();
}
