#include "rheology.hpp"
#include <algorithm>
#include <cmath>

namespace fld::rheology
{
    float viscosity(const material &m, float shear_rate)
    {
        const float high = std::max(.000001f, m.viscosity_infinite);
        const float low = std::max(high, m.viscosity_zero);
        const float x = std::max(0.f, m.relaxation * shear_rate);
        return high + (low - high) *
            std::pow(1.f + x*x, (m.flow_index - 1.f) * .5f);
    }

    float reynolds(const material &m, float speed, float diameter)
    {
        return m.density * std::abs(speed) * std::max(0.f, diameter) /
               viscosity(m, speed / std::max(.0001f, diameter));
    }

    float weber(const material &m, float speed, float diameter)
    {
        return m.density * speed * speed * std::max(0.f, diameter) /
               std::max(.000001f, m.surface_tension);
    }
}
