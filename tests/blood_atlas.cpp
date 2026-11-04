#include "../blood/bld/surfaces/stains.hpp"

#include <cassert>
#include <cstdint>

namespace
{
    std::uint32_t coverage(const bld::surface::atlas &a,bld::vec2 uv)
    {
        const auto x=std::uint32_t(uv.x*float(a.resolution-1));
        const auto y=std::uint32_t(uv.y*float(a.resolution-1));
        return a.pixels[std::size_t(y)*a.resolution+x]>>24u;
    }
}

int main()
{
    bld::options options{};
    options.wound_resolution=64;
    options.max_marks=1; // Force flattening on the second impact.
    bld::surface::init(options);
    bld::surface::register_target(
        7,bld::substrate::textile,bld::mat4::identity(),
        true,bld::vec2{.5f,.5f});

    const bld::vec2 first{.2f,.2f},second{.8f,.8f};
    bld::surface::deposit(7,{0,0,0},{0,0,1},
                          first,true,1.f,2.f,1);
    const auto *a=bld::surface::find(7);
    assert(a && coverage(*a,first)>0);

    bld::surface::deposit(7,{0,0,0},{0,0,1},
                          second,true,1.f,2.f,2);
    a=bld::surface::find(7);
    assert(a && a->marks.size()==1);
    assert(a->baked_pixels.size()==a->pixels.size());
    assert(coverage(*a,first)>0 && coverage(*a,second)>0);

    const std::uint32_t x=std::uint32_t(first.x*(a->resolution-1));
    const std::uint32_t y=std::uint32_t(first.y*(a->resolution-1));
    const auto i=std::size_t(y)*a->resolution+x;
    const auto before=(a->baked_pixels[i]>>16u)&255u;
    assert(before>0);
    bld::surface::update(20.f);
    a=bld::surface::find(7);
    assert(((a->baked_pixels[i]>>16u)&255u)<before);
    assert(coverage(*a,first)>0 && coverage(*a,second)>0);

    bld::surface::shutdown();
    return 0;
}
