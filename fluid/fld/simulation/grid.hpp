#pragma once
#include "../core/types.hpp"
#include <cstdint>
#include <utility>
#include <vector>

namespace fld::simulation
{
    struct pair
    {
        std::uint32_t a, b;
    };

    // Stable neighbor enumeration with cell size equal to kernel support.
    // Rebuilt following each positional solve iteration to prevent stale
    // neighbors when particle density constraints move the particles.
    std::vector<pair> neighbors(const std::vector<particle> &particles,
                                float smoothing_radius);
}
