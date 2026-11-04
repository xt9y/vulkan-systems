#include "geometry.hpp"
#include <algorithm>
#include <cmath>

namespace bld::math
{
    mat4 inverse_affine(const mat4 &m)
    {
        using namespace vk::util::math;
        const vec3 a{m(0, 0), m(1, 0), m(2, 0)};
        const vec3 b{m(0, 1), m(1, 1), m(2, 1)};
        const vec3 c{m(0, 2), m(1, 2), m(2, 2)};
        const float det = dot(a, cross(b, c));
        if (std::abs(det) < 1e-10f) return mat4::identity();
        const vec3 x = cross(b, c) / det;
        const vec3 y = cross(c, a) / det;
        const vec3 z = cross(a, b) / det;
        mat4 result = mat4::identity();
        for (int i = 0; i < 3; ++i)
        {
            result(0, i) = (&x.x)[i];
            result(1, i) = (&y.x)[i];
            result(2, i) = (&z.x)[i];
        }
        const vec3 p{m(0, 3), m(1, 3), m(2, 3)};
        result(0, 3) = -dot(x, p);
        result(1, 3) = -dot(y, p);
        result(2, 3) = -dot(z, p);
        return result;
    }

    vec3 direction(vec3 value, vec3 fallback)
    {
        using namespace vk::util::math;
        return length_squared(value) > 1e-12f ? normalize(value) : fallback;
    }

    vec3 transform_point(const mat4 &m, vec3 p)
    {
        const auto v = m * vec4{p, 1};
        return {v.x, v.y, v.z};
    }

    vec3 position(const mat4 &m)
    {
        return {m(0, 3), m(1, 3), m(2, 3)};
    }

    mat4 projector(vec3 point, vec3 normal, float width, float height, float rotation)
    {
        using namespace vk::util::math;
        const vec3 n = direction(normal);
        const vec3 guide = std::abs(n.y) < .95f ? vec3{0, 1, 0} : vec3{1, 0, 0};
        const vec3 a = direction(cross(guide, n), {1, 0, 0});
        const vec3 b = cross(n, a);
        const auto x = a * std::cos(rotation) + b * std::sin(rotation);
        const auto y = b * std::cos(rotation) - a * std::sin(rotation);
        mat4 out = mat4::identity();
        const float depth = std::max(.10f, std::max(width, height) * .3f);
        for (int i = 0; i < 3; ++i)
        {
            out(i, 0) = (&x.x)[i] * width;
            out(i, 1) = (&y.x)[i] * height;
            out(i, 2) = (&n.x)[i] * depth;
        }
        out(0, 3) = point.x + n.x * .001f;
        out(1, 3) = point.y + n.y * .001f;
        out(2, 3) = point.z + n.z * .001f;
        return out;
    }

    std::uint32_t random(std::uint32_t &seed)
    {
        if (!seed) seed = 1;
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        return seed;
    }

    float unit(std::uint32_t &seed)
    {
        return float(random(seed) & 0xffffffu) * (1.0f / 16777216.0f);
    }
}
