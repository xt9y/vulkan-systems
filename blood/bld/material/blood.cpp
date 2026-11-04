#include "blood.hpp"
#include <algorithm>

namespace bld::material
{
    fld::material physical()
    {
        fld::material blood{};
        blood.density=1055.f;
        blood.viscosity_zero=.0082f;
        blood.viscosity_infinite=.0035f;
        blood.relaxation=.1138f;
        blood.flow_index=.62f;
        blood.surface_tension=.05f;
        blood.contact_angle=1.05f;
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
