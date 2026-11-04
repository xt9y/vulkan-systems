#include "contact.hpp"
#include "rheology.hpp"
#include "../math/kernels.hpp"
#include <algorithm>
#include <cmath>

namespace fld::wetting
{
    diagnostics resolve(const material &fluid, const particle &drop,
                        collision &hit)
    {
        const auto &surface=hit.surface;
        const float size=std::max(.0001f,drop.radius*2.f);
        const float speed=math::norm(drop.velocity);
        diagnostics d{};
        d.reynolds=rheology::reynolds(fluid,speed,size);
        d.weber=rheology::weber(fluid,speed,size);
        d.ohnesorge=std::sqrt(d.weber)/std::max(d.reynolds,.00001f);
        d.splash_parameter=std::sqrt(d.weber)*
                           std::pow(std::max(0.f,d.reynolds),.25f);
        const float porosity=std::clamp(surface.porosity,0.f,1.f);
        const float wetting=std::clamp(surface.wetting,0.f,1.f);
        const float contact_angle=std::clamp(
            surface.contact_angle,0.f,3.14159265f);
        const float adhesion=std::clamp(
            .5f*(1.f+std::cos(contact_angle))*.55f + wetting*.45f,
            0.f,1.f);
        // Dimensionless impact breakup parameter after Mundo et al.
        // Roughness lowers the nominal breakup threshold, while porous
        // substrates capture a significant fraction of the lamella.
        const float threshold=57.7f/
            (1.f+std::clamp(surface.roughness,0.f,1.f)*1.8f);
        d.splash=d.splash_parameter>threshold && porosity<.55f;
        const float stick=std::clamp(
            adhesion*.60f + porosity*.85f + .14f, .02f,1.f);
        const float breakup=d.splash?
            std::clamp((d.splash_parameter-threshold)/
                       std::max(threshold,.01f),0.f,1.f):0.f;
        hit.deposition_fraction=std::clamp(
            stick*(1.f-.72f*breakup),.015f,1.f);
        hit.absorb=porosity>.75f ||
            (!d.splash && stick>.89f && speed<1.f);
        if(hit.absorb)hit.deposition_fraction=1.f;
        hit.restitution=std::clamp(
            surface.restitution*(1.f-adhesion) +
            (d.splash?.09f:0.f),0.f,.55f);
        hit.friction=std::clamp(
            surface.friction + adhesion*.20f + porosity*.25f,0.f,1.f);
        return d;
    }
}
