#include "properties.hpp"
#include <cmath>
#include <stdexcept>

namespace fld
{
    bool valid(const properties &p)
    {
        constexpr float pi=3.14159265358979323846f;
        return std::isfinite(p.density) && p.density>0.f &&
               std::isfinite(p.dynamic_viscosity) &&
               p.dynamic_viscosity>0.f &&
               std::isfinite(p.surface_tension) && p.surface_tension>0.f &&
               std::isfinite(p.advancing_angle) &&
               std::isfinite(p.receding_angle) &&
               p.receding_angle>=0.f &&
               p.advancing_angle>=p.receding_angle &&
               p.advancing_angle<=pi &&
               std::isfinite(p.index_of_refraction) &&
               p.index_of_refraction>=1.f;
    }

    static void require(const properties &p)
    {
        if(!valid(p))throw std::invalid_argument("Invalid fluid properties");
    }
    static void positive(float x)
    {
        if(!std::isfinite(x) || x<=0.f)
            throw std::invalid_argument("Positive finite scale required");
    }

    float capillary_length(const properties &p,float gravity)
    {
        require(p);positive(gravity);
        return std::sqrt(p.surface_tension/(p.density*gravity));
    }
    float laplace_pressure(const properties &p,float radius)
    {
        require(p);positive(radius);
        return 2.f*p.surface_tension/radius;
    }
    float bond_number(const properties &p,float size,float gravity)
    {
        require(p);positive(size);positive(gravity);
        return p.density*gravity*size*size/p.surface_tension;
    }
    float weber_number(const properties &p,float speed,float size)
    {
        require(p);positive(size);
        if(!std::isfinite(speed))throw std::invalid_argument("Bad speed");
        return p.density*speed*speed*size/p.surface_tension;
    }
    float ohnesorge_number(const properties &p,float size)
    {
        require(p);positive(size);
        return p.dynamic_viscosity/
               std::sqrt(p.density*p.surface_tension*size);
    }
    float pinning_force(const properties &p)
    {
        require(p);
        return p.surface_tension*
            (std::cos(p.receding_angle)-std::cos(p.advancing_angle));
    }
}
