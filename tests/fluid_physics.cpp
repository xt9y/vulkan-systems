#include "../fluid/fld/core/system.hpp"
#include "../fluid/fld/material/rheology.hpp"
#include "../fluid/fld/material/contact.hpp"
#include "../fluid/fld/math/kernels.hpp"

#include <cassert>
#include <cmath>
#include <iostream>

int main()
{
    // Deterministic smoke test: blood and water share identical code.
    fld::options settings{};
    settings.spacing=.004f;
    settings.smoothing_radius=.010f;
    settings.max_particles=4096;
    fld::system::init(settings);

    fld::material blood{};
    blood.density=1055.f;
    blood.viscosity_zero=.160f;
    blood.viscosity_infinite=.0035f;
    blood.relaxation=8.2f;
    blood.flow_index=.2128f;
    blood.yasuda_exponent=.64f;
    blood.surface_tension=.05f;
    auto phase=fld::system::add_material(blood);

    assert(fld::rheology::viscosity(blood,0.f)>
           fld::rheology::viscosity(blood,1000.f));
    assert(fld::rheology::viscosity(blood,1000.f)>=
           blood.viscosity_infinite);

    fld::emission source{};
    source.phase=phase;
    source.position={0,.12f,0};
    source.velocity={0,-.1f,0};
    source.volume=.000002f; // 2mL
    const auto count=fld::system::emit(source);
    assert(count>0);

    const double mass=source.volume*blood.density;
    auto* state=fld::system::get();
    assert(std::abs(state->stats.live_mass_kg-mass)<mass*.0001);

    fld::collision_callback ground=[](fld::vec3 from,fld::vec3 to,float radius)
    {
        fld::collision contact{};
        if(to.y>radius || from.y<radius)return contact;
        contact.hit=true;
        contact.target=1;
        contact.position={to.x,0,to.z};
        contact.normal={0,1,0};
        contact.surface.porosity=1.f;
        contact.surface.wetting=1.f;
        return contact;
    };

    double deposited=0;
    fld::contact_callback collection=[&](const fld::particle &particle,
                                          const fld::collision &hit)
    {
        if(hit.target==1) deposited+=particle.mass;
    };

    for(int step=0;step<90;++step)
        fld::system::update(1.f/90.f,ground,collection);

    assert(deposited>0.);
    assert(state->stats.deposited_mass_kg>0.);
    const double balance=state->stats.deposited_mass_kg+
                         state->stats.live_mass_kg+
                         state->stats.expired_mass_kg;
    const double relative=std::abs(balance-mass)/mass;
    assert(relative<.025);  // bounded solver mass conservation.

    // The generic impact API must work with an arbitrary liquid.
    fld::splash_source splash{};
    splash.phase=phase;
    splash.position={0,.4f,0};
    splash.normal={0,1,0};
    splash.velocity={0,4,0};
    splash.volume=.000001f;
    splash.group=42;
    splash.surface.porosity=.25f;
    const auto result=fld::system::splash(splash);
    assert(result.particles>0);
    assert(result.weber>0);
    assert(result.reynolds>0);
    assert(std::abs(result.retained_volume+result.emitted_volume-
                    splash.volume)<splash.volume*.0001f);

    fld::system::clear_phase(phase);
    for(const auto &particle:fld::system::get()->particles)
        assert(particle.phase!=phase);
    fld::system::shutdown();
    std::cout << "fluid physics passed, relative mass error: "
              << relative << "\n";
    return 0;
}
