#pragma once

#include "../core/types.hpp"
#include <cstdint>

namespace bld::wounds
{
    // A wound is a bounded, supplied volume reservoir. The game chooses
    // amount and flow parameters; no anatomical injury is inferred solely
    // from a bullet contact point.
    struct source
    {
        std::uint64_t id=0;
        instance_id target=0;
        vec3 position{},normal{0,1,0};
        vec2 uv{};
        bool uv_valid=false;
        float remaining_ml=0.f;
        float initial_rate_ml_s=0.f;
        float decay_seconds=8.f;
        float pulse_hz=0.f;
        float age=0.f;
        std::uint32_t seed=0;
    };

    // Returns the exact volume withdrawn from the reservoir in millilitres.
    float advance(source &wound,float dt);
}
