#include "blood.hpp"
#include <algorithm>
#include <cmath>

namespace bld::material
{
    fld::material physical()
    {
        fld::material blood{};
        blood.density=1055.f;
        // Published human-blood Carreau-Yasuda fit (ETH 2020, section
        // 4.1): mu0=.16 Pa*s, mu_infinity=.0035 Pa*s, lambda=8.2 s,
        // inner exponent=.64, outer exponent=-1.23.
        // Slightly stronger low-/high-shear viscosity for coherent sheets.
        blood.viscosity_zero=.220f;
        blood.viscosity_infinite=.0045f;
        blood.relaxation=8.2f;
        blood.flow_index=.2128f;
        blood.yasuda_exponent=.64f;
        blood.surface_tension=.058f;
        blood.contact_angle=1.05f;
        blood.optical=optical();
        return blood;
    }

    fld::render::appearance optical(float oxygenation)
    {
        const float oxygen=std::clamp(oxygenation,0.f,1.f);
        fld::render::appearance blood{};
        // RGB absorption is an ARTIST-CALIBRATED approximation, not a
        // substitute for measured wavelength-dependent hemoglobin spectra.
        blood.absorption={
            125.f-20.f*oxygen,
            610.f+105.f*oxygen,
            475.f+110.f*oxygen
        };
        blood.scattering={.055f,.013f,.012f};
        blood.index_of_refraction=1.35f;
        blood.roughness=.13f;
        return blood;
    }

    fld::surface_properties interaction(substrate kind)
    {
        const auto visual=surface(kind);
        fld::surface_properties response{};
        response.porosity=visual.absorption;
        response.contact_angle=visual.contact_angle;
        response.wetting=std::clamp(1.f-
            visual.contact_angle/3.14159265f,.05f,.95f);
        response.roughness=std::clamp(visual.roughness,0.f,1.f);
        response.friction=std::clamp(.18f+visual.absorption*.63f,0.f,1.f);
        response.restitution=std::clamp(.15f-
            visual.absorption*.12f,.01f,.15f);
        return response;
    }

    substrate_properties surface(substrate kind)
    {
        switch(kind)
        {
            case substrate::skin:
                return {.12f,.22f,1.02f,.95f,1.05f,1.0f};
            case substrate::textile:
                return {.92f,.85f,1.6f,.18f,.38f,.55f};
            case substrate::porous:
                return {.74f,.89f,1.4f,.30f,.58f,.72f};
            case substrate::painted:
                return {.12f,.30f,1.08f,.85f,1.04f,1.25f};
            case substrate::metal:
                return {.01f,.22f,.88f,.97f,1.3f,1.7f};
            case substrate::glass:
                return {.0f,.08f,.73f,.99f,1.45f,2.0f};
        }
        return {};
    }
}
