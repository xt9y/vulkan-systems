#pragma once

// Conservative mobile-film -> pore uptake and in-plane capillary transport.
// All volumes are cubic metres and areas are square metres. This is a
// homogenized textile/porous-material model, not individual fibres.
namespace fld::porous
{
    struct properties
    {
        float capacity_m = 0.f;  // pore capacity as equivalent film depth
        float uptake_m_s = 0.f;  // infiltration rate while unsaturated
        float wick_u_m2_s = 0.f;
        float wick_v_m2_s = 0.f;
    };

    // Transfer volume between coupled reservoirs without inventing fluid.
    float absorb(float &mobile_m3,float &stored_m3,float area_m2,
                 float seconds,const properties &material);

    // Signed volume transferred from a to b over one timestep. Caller
    // applies this to both reservoirs simultaneously; neither goes negative.
    float capillary_flux(float stored_a,float stored_b,
                         float area_m2,float distance_m,float seconds,
                         float diffusivity_m2_s,float capacity_m3);
}
