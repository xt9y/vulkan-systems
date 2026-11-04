#include "../blood/bld/core/system.hpp"
#include "../blood/bld/surfaces/stains.hpp"
#include <fld/core/system.hpp>
#include <cassert>
#include <cmath>

int main()
{
    bld::options options{};
    options.wound_resolution=64;
    options.fluid.max_particles=4096;
    bld::system::init(options);
    bld::surface::register_target(5,bld::substrate::skin,
        bld::mat4::identity(),true,{.4f,.4f});

    bld::impact wound{};
    wound.target=5;
    wound.position={0,.5f,0};
    wound.normal={0,0,1};
    wound.direction={0,0,1};
    wound.uv={.5f,.4f};
    wound.uv_valid=true;
    wound.amount_ml=2.f;
    wound.velocity=2.f;
    wound.bleeding_reservoir_ml=4.f;
    wound.bleeding_rate_ml_s=1.5f;
    wound.bleeding_decay_seconds=10.f;
    bld::system::hit(wound);
    assert(bld::system::get()->open_wounds.size()==1);
    const double initial_surface=bld::surface::volume();
    assert(initial_surface>0.);

    for(int i=0;i<120;++i)
        bld::system::update(1.f/60.f);
    assert(bld::system::get()->wound_supplied_ml>0.);
    assert(bld::system::get()->wound_supplied_ml<=4.);
    assert(bld::surface::volume()>=initial_surface);
    double remaining=0;
    for(const auto &s:bld::system::get()->open_wounds)
        remaining+=s.remaining_ml;
    assert(std::abs(remaining+
        bld::system::get()->wound_supplied_ml-4.)<1.e-3);

    bld::system::shutdown();
    assert(!bld::system::get());
}
