#include "grid.hpp"
#include <cmath>
#include <unordered_map>
#include <cstddef>

namespace fld::simulation
{
    namespace
    {
        struct cell
        {
            int x, y, z;
            bool operator==(const cell &other) const
            {
                return x==other.x && y==other.y && z==other.z;
            }
        };
        struct hash
        {
            std::size_t operator()(const cell &c) const noexcept
            {
                const auto x=std::uint64_t(std::uint32_t(c.x))*73856093ull;
                const auto y=std::uint64_t(std::uint32_t(c.y))*19349663ull;
                const auto z=std::uint64_t(std::uint32_t(c.z))*83492791ull;
                return std::size_t(x ^ y ^ z);
            }
        };

        cell index(vec3 point, float h)
        {
            return {
                int(std::floor(point.x/h)),
                int(std::floor(point.y/h)),
                int(std::floor(point.z/h))
            };
        }
    }

    std::vector<pair> neighbors(const std::vector<particle> &ps, float h)
    {
        std::unordered_map<cell,std::vector<std::uint32_t>,hash> cells;
        cells.reserve(ps.size());
        for (std::uint32_t i=0; i<ps.size(); ++i)
            cells[index(ps[i].predicted,h)].push_back(i);
        std::vector<pair> result;
        result.reserve(ps.size()*24);
        for (std::uint32_t a=0; a<ps.size(); ++a)
        {
            const auto c=index(ps[a].predicted,h);
            for (int z=-1; z<=1; ++z)
                for (int y=-1; y<=1; ++y)
                    for (int x=-1; x<=1; ++x)
                    {
                        const auto it=cells.find({c.x+x,c.y+y,c.z+z});
                        if(it==cells.end())continue;
                        for(auto b:it->second)
                            if(b>a && ps[a].phase == ps[b].phase &&
                               vk::util::math::length_squared(
                                   ps[a].predicted-ps[b].predicted)<h*h)
                                result.push_back({a,b});
                    }
        }
        return result;
    }
}
