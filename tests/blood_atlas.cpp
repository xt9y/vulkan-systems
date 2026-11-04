#include "../blood/bld/surfaces/stains.hpp"
#include <cassert>
#include <cmath>

int main()
{
    bld::options options{};
    options.wound_resolution=64;
    options.max_marks=1; // History eviction must not erase liquid.
    bld::surface::init(options);
    bld::surface::register_target(7,bld::substrate::textile,
        bld::mat4::identity(),true,bld::vec2{.5f,.5f});

    const bld::vec2 first{.2f,.2f},second{.8f,.8f};
    bld::surface::deposit(7,{0,0,0},{0,0,1},first,true,1.f,2.f,1);
    auto *a=bld::surface::find(7);
    assert(a && a->film.cells.size()==64u*64u);
    assert(a->pixels.size()==a->film.cells.size());
    assert(std::abs(bld::surface::volume()-1.e-6)<1.e-10);

    bld::surface::deposit(7,{0,0,0},{0,0,1},second,true,1.f,2.f,2);
    a=bld::surface::find(7);
    assert(a && a->marks.size()==1);
    assert(std::abs(bld::surface::volume()-2.e-6)<2.e-10);

    for(int i=0;i<120;++i)
        bld::surface::update(1.f/60.f);
    a=bld::surface::find(7);
    assert(a && a->pixels.size()==64u*64u);
    // Model transfers mass into pores without arbitrarily deleting it.
    double absorbed=0;
    for(const auto &cell:a->film.cells)absorbed+=cell.stored;
    assert(absorbed>0.);
    assert(std::abs(bld::surface::volume()-2.e-6)<2.e-9);

    bld::surface::remove_target(7);
    assert(bld::surface::volume()==0.);
    bld::surface::shutdown();
}
