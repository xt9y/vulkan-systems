#include "geometry.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace bld::math
{
    mat4 inverse(const mat4 &m)
    {
        double rows[4][8]{};
        for (int row = 0; row < 4; ++row)
        {
            for (int col = 0; col < 4; ++col)
                rows[row][col] = m(row, col);
            rows[row][row + 4] = 1.0;
        }
        for (int col = 0; col < 4; ++col)
        {
            int pivot = col;
            for (int row = col + 1; row < 4; ++row)
                if (std::abs(rows[row][col]) > std::abs(rows[pivot][col]))
                    pivot = row;
            if (std::abs(rows[pivot][col]) < 1e-12)
                throw std::runtime_error("Cannot invert singular matrix");
            if (col != pivot)
                for (int n = 0; n < 8; ++n)
                    std::swap(rows[col][n], rows[pivot][n]);
            const double d = rows[col][col];
            for (int n = 0; n < 8; ++n) rows[col][n] /= d;
            for (int row = 0; row < 4; ++row)
            {
                if (row == col) continue;
                const double factor = rows[row][col];
                for (int n = 0; n < 8; ++n)
                    rows[row][n] -= factor * rows[col][n];
            }
        }
        mat4 result{};
        for (int row = 0; row < 4; ++row)
            for (int col = 0; col < 4; ++col)
                result(row, col) = float(rows[row][col + 4]);
        return result;
    }

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
        // A thin surface layer, not a volume extending through body parts.
        const float depth = std::clamp(std::max(width, height) * .20f, .008f, .04f);
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
