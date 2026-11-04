#pragma once
#include "../core/types.hpp"

namespace fld::rheology
{
    // Carreau-Yasuda shear-thinning approximation. Constant-viscosity
    // materials are represented by viscosity_zero == viscosity_infinite.
    float viscosity(const material &properties, float shear_rate);
    float reynolds(const material &properties, float speed, float diameter);
    float weber(const material &properties, float speed, float diameter);
}
