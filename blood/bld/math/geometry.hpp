#pragma once
#include "../core/types.hpp"

namespace bld::math
{
    mat4 inverse_affine(const mat4 &matrix);
    mat4 projector(vec3 point, vec3 normal, float width, float height, float rotation = 0);
    vec3 position(const mat4 &matrix);
    vec3 transform_point(const mat4 &matrix, vec3 point);
    vec3 direction(vec3 value, vec3 fallback = {0, 1, 0});
    std::uint32_t random(std::uint32_t &seed);
    float unit(std::uint32_t &seed);
}
