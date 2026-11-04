#pragma once
#include "types.hpp"

namespace fld::system
{
    struct state
    {
        options settings{};
        std::vector<material> materials;
        std::vector<particle> particles;
        statistics stats{};
        double remainder = 0;
    };

    state *get();
    void init(const options &settings = {});
    std::uint32_t add_material(const fld::material &properties);
    std::uint32_t emit(const fld::emission &source);
    void update(float seconds,
                const collision_callback &collision = {},
                const contact_callback &contact = {});
    void clear();
    void shutdown();
}
