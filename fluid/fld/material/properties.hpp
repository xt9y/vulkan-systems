#pragma once

namespace fld
{
    // SI units: kg/m^3, Pa*s, N/m and radians. Do not substitute
    // dimensionless viscosity sliders for physical model parameters.
    struct properties
    {
        float density=998.f;
        float dynamic_viscosity=.001f;
        float surface_tension=.072f;
        float advancing_angle=1.3f;
        float receding_angle=1.0f;
        float index_of_refraction=1.333f;
    };

    bool valid(const properties &liquid);
    float capillary_length(const properties &liquid,float gravity=9.81f);
    float laplace_pressure(const properties &liquid,float radius);
    float bond_number(const properties &liquid,float size,float gravity=9.81f);
    float weber_number(const properties &liquid,float speed,float size);
    float ohnesorge_number(const properties &liquid,float size);
    // Magnitude of contact-line pinning force per unit length (N/m).
    float pinning_force(const properties &liquid);
}
