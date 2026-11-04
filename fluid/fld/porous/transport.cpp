#include "transport.hpp"
#include <algorithm>
#include <cmath>

namespace fld::porous
{
    float absorb(float &mobile,float &stored,float area,float dt,
                 const properties &m)
    {
        if(!(area>0.f && dt>0.f && std::isfinite(mobile) &&
             std::isfinite(stored) && mobile>0.f))return 0.f;
        const float capacity=std::max(0.f,m.capacity_m)*area;
        const float available=std::max(0.f,capacity-stored);
        const float amount=std::min({mobile,available,
            std::max(0.f,m.uptake_m_s)*area*dt});
        mobile-=amount;
        stored+=amount;
        return amount;
    }

    float capillary_flux(float a,float b,float area,float dx,float dt,
                         float diffusivity,float capacity)
    {
        if(!(area>0 && dx>0 && dt>0 && diffusivity>0))return 0.f;
        // Diffusion in saturation (not metres of liquid): D dt / dx^2
        // is dimensionless. A signed pair update conserves pore volume.
        const float amount=std::clamp(diffusivity*dt/(dx*dx),0.f,.24f)
                           *(a-b);
        if(amount>0.f)
            return std::min({amount,std::max(0.f,a),
                             std::max(0.f,capacity-b)});
        return -std::min({-amount,std::max(0.f,b),
                          std::max(0.f,capacity-a)});
    }
}
