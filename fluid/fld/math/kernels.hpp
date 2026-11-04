#pragma once
#include "../core/types.hpp"

namespace fld::math
{
    float poly6(float distance_squared, float smoothing_radius);
    vec3 spiky_gradient(vec3 separation, float smoothing_radius);
    float viscosity_laplacian(float distance, float smoothing_radius);
    float norm(vec3 value);
    vec3 safe_normal(vec3 value);
    std::uint32_t random(std::uint32_t &seed);
    float random01(std::uint32_t &seed);
}
