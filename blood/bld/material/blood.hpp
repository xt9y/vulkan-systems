#pragma once
#include "../core/types.hpp"

namespace bld::material
{
    // Parameters are SI measurements and bounded calibration values.
    fld::material physical();
    fld::render::appearance optical(float oxygenation=.95f);
    struct substrate_properties
    {
        float absorption=0, roughness=.15f;
        float spread=1, wetness=1;
        float contact_angle=1.1f; // rad
        float drying_scale=1;
    };
    substrate_properties surface(substrate kind);
}
