#pragma once
#include <vs/scene/scene.hpp>
#include <array>
#include <cmath>
#include <algorithm>
#include <cstdint>

namespace vs::gpu {
// Shared shader ABI for point lights in PCloud, Optics and user shaders.
// Keeps std430 bytes identical without duplicated struct/packing code.
struct alignas(16) point_light {
    std::array<float,4> position_range{};
    std::array<float,4> color_intensity{};
};
static_assert(sizeof(point_light)==32);

inline point_light pack(const ::vs::scene::point_light& value) {
    return {{
        value.position.x,value.position.y,value.position.z,value.radius
    },{
        value.color.x,value.color.y,value.color.z,
        value.enabled?value.strength:0.f
    }};
}

// Compact, independent light-shape ABI, shared by PCloud, glass and optics.
// The original 32-byte point-light ABI is unchanged for other VS renderers.
struct alignas(16) light_shape {
    std::array<float,4> direction_cutoff{};
};
static_assert(sizeof(light_shape)==16);
inline light_shape pack_shape(const ::vs::scene::point_light& value) {
    if(value.kind!=::vs::scene::light_kind::slight)
        return {{{0.f,-1.f,0.f,-1.f}}};
    const auto d=value.direction;
    const float length=std::sqrt(d.x*d.x+d.y*d.y+d.z*d.z);
    const float inv=length>1.e-6f?1.f/length:0.f;
    const float angle=std::clamp(value.cone_degrees,1.f,179.f)*
        .008726646259971648f;
    return {{{length>1.e-6f?d.x*inv:0.f,
              length>1.e-6f?d.y*inv:-1.f,
              length>1.e-6f?d.z*inv:0.f,std::cos(angle)}}};
}
}
