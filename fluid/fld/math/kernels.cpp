#include "kernels.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace fld::math
{
    namespace { constexpr float pi = 3.14159265358979323846f; }

    float poly6(float distance_squared, float h)
    {
        if (distance_squared >= h*h || h <= 0.f) return 0.f;
        const float q = h*h - distance_squared;
        return 315.f / (64.f * pi * std::pow(h, 9.f)) * q*q*q;
    }

    vec3 spiky_gradient(vec3 displacement, float h)
    {
        const float r2 = vk::util::math::length_squared(displacement);
        if (h <= 0.f || r2 <= 1e-12f || r2 >= h*h) return {};
        const float r = std::sqrt(r2);
        const float x = h-r;
        return displacement * (-45.f / (pi * std::pow(h, 6.f)) * x*x / r);
    }

    float viscosity_laplacian(float r, float h)
    {
        if (r >= h || h <= 0.f) return 0.f;
        return 45.f / (pi * std::pow(h, 6.f)) * (h - r);
    }

    float norm(vec3 v)
    {
        return std::sqrt(vk::util::math::length_squared(v));
    }

    vec3 safe_normal(vec3 v)
    {
        const auto r = norm(v);
        return r > 1e-9f ? v / r : vec3{0, 1, 0};
    }

    mat4 inverse(const mat4 &matrix)
    {
        double rows[4][8]{};
        for(int row=0;row<4;++row)
        {
            for(int col=0;col<4;++col)
                rows[row][col]=matrix(row,col);
            rows[row][row+4]=1.0;
        }
        for(int col=0;col<4;++col)
        {
            int pivot=col;
            for(int row=col+1;row<4;++row)
                if(std::abs(rows[row][col])>std::abs(rows[pivot][col]))
                    pivot=row;
            if(std::abs(rows[pivot][col])<1e-12)
                throw std::invalid_argument("Singular view-projection matrix");
            if(pivot!=col)
                for(int c=0;c<8;++c)
                    std::swap(rows[col][c],rows[pivot][c]);
            const double coefficient=rows[col][col];
            for(int c=0;c<8;++c)rows[col][c]/=coefficient;
            for(int row=0;row<4;++row)
            {
                if(row==col)continue;
                const double coefficient=rows[row][col];
                for(int c=0;c<8;++c)
                    rows[row][c]-=coefficient*rows[col][c];
            }
        }
        mat4 inverse{};
        for(int row=0;row<4;++row)
            for(int col=0;col<4;++col)
                inverse(row,col)=float(rows[row][col+4]);
        return inverse;
    }

    std::uint32_t random(std::uint32_t &seed)
    {
        if (!seed) seed = 1;
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        return seed;
    }

    float random01(std::uint32_t &seed)
    {
        return float(random(seed) & 0x00ffffffu) / float(0x01000000u);
    }
}
