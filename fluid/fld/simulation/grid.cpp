#include "grid.hpp"
#include <cmath>
#include <map>

namespace fld::simulation
{
    namespace
    {
        struct cell
        {
            int x, y, z;
            bool operator<(const cell &other) const
            {
                if (x != other.x) return x < other.x;
                if (y != other.y) return y < other.y;
                return z < other.z;
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
        std::map<cell, std::vector<std::uint32_t>> cells;
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
