#include "../fluid/fld/material/properties.hpp"
#include <cassert>
#include <cmath>
#include <stdexcept>

int main()
{
    fld::properties water{};
    assert(fld::valid(water));
    const float capillary=fld::capillary_length(water);
    assert(capillary>.0026f && capillary<.0028f);
    const float small=fld::laplace_pressure(water,.00025f);
    const float large=fld::laplace_pressure(water,.001f);
    assert(std::abs(small/large-4.f)<1e-5f);
    assert(fld::bond_number(water,.0005f)<
           fld::bond_number(water,.005f));
    assert(fld::weber_number(water,0.f,.001f)==0.f);
    assert(fld::ohnesorge_number(water,.001f)>0.f);
    assert(fld::pinning_force(water)>0.f);
    water.receding_angle=2.f;
    assert(!fld::valid(water));
    bool rejected=false;
    try{fld::laplace_pressure(water,0.f);}
    catch(const std::invalid_argument &){rejected=true;}
    assert(rejected);
}
