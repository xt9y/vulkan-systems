#include "../fluid/fld/surface/film.hpp"
#include "../fluid/fld/porous/transport.hpp"
#include "../blood/bld/wounds/flow.hpp"
#include <cassert>
#include <cmath>

static double center_y(const fld::surface::grid &f)
{
    double mass=0,moment=0;
    for(std::uint32_t y=0;y<f.height;++y)
        for(std::uint32_t x=0;x<f.width;++x)
        {
            const auto &c=f.cells[std::size_t(y)*f.width+x];
            mass+=c.mobile;
            moment+=double(y)*c.mobile;
        }
    return mass>0.?moment/mass:0.;
}

int main()
{
    fld::material blood{};
    blood.density=1055.f;
    blood.viscosity_zero=.22f;
    blood.viscosity_infinite=.0045f;
    blood.relaxation=8.2f;
    blood.yasuda_exponent=.64f;
    blood.flow_index=.2128f;
    blood.surface_tension=.058f;

    fld::surface::grid film;
    fld::surface::init(film,48,48,{.12f,.12f});
    fld::surface::set_frame(film,{0,0,1},fld::mat4::identity());
    fld::surface::deposit(film,{.5f,.25f},.014f,2.e-6f);
    const auto initial_y=center_y(film);
    const auto before=fld::surface::volume(film);
    assert(std::abs(before-2.e-6)<1.e-10);
    for(int i=0;i<60;++i)
        fld::surface::update(film,1.f/60.f,blood,{});
    assert(std::abs(fld::surface::volume(film)-before)<1.e-9);
    assert(center_y(film)>initial_y);

    fld::surface::grid cloth;
    fld::surface::init(cloth,48,48,{.12f,.12f});
    fld::surface::set_frame(cloth,{0,0,1},fld::mat4::identity());
    fld::surface::deposit(cloth,{.5f,.5f},.014f,1.e-6f);
    fld::porous::properties fibres{.0016f,.00030f,.0000016f,.00000045f};
    for(int i=0;i<60;++i)
        fld::surface::update(cloth,1.f/60.f,blood,fibres);
    double soaked=0.;
    for(const auto &c:cloth.cells)soaked+=c.stored;
    assert(soaked>0.);
    assert(std::abs(fld::surface::volume(cloth)-1.e-6)<1.e-9);

    const auto index=std::size_t(12)*film.width+24;
    const float available=film.cells[index].mobile;
    if(available>0.f)
    {
        const float detached=available*.2f;
        assert(fld::surface::take(film,24,12,detached));
        assert(std::abs(fld::surface::volume(film)+detached-before)<1.e-9);
    }

    bld::wounds::source wound{};
    wound.remaining_ml=9.f;
    wound.initial_rate_ml_s=2.f;
    wound.decay_seconds=5.f;
    float delivered=0.f;
    for(int i=0;i<600;++i)
        delivered+=bld::wounds::advance(wound,1.f/60.f);
    assert(delivered>0.f && delivered<=9.f);
    assert(std::abs(delivered+wound.remaining_ml-9.f)<1.e-3f);
}
