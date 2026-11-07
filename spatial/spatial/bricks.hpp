#pragma once
#include "index.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace spatial
{
    // Compact, deterministic candidate lists for regularly spaced 3D queries.
    // One list serves brick_size^3 adjacent query positions. The consumer
    // still performs its exact distance/visibility tests on each candidate.
    // Candidate IDs retain source order, preserving floating-point reduction
    // order for kernels that previously visited every source element.
    struct bricked_neighborhood
    {
        std::vector<std::uint32_t> offsets;
        std::vector<std::uint32_t> indices;
        std::array<std::uint32_t,3> bricks{1,1,1};
        std::uint32_t brick_size=4;

        // A UINT32_MAX first offset means the index exceeded its memory
        // budget. Kernels should fall back to scanning all input points.
        bool indexed() const noexcept
        {
            return !offsets.empty() &&
                   offsets.front()!=std::numeric_limits<std::uint32_t>::max();
        }
    };

    inline bricked_neighborhood build_bricked_neighborhood(
        const std::vector<vec3>& positions,vec3 origin,float spacing,
        std::array<std::uint32_t,3> dimensions,float radius,
        std::uint32_t brick_size=4,std::size_t max_indices=8000000)
    {
        if(!brick_size || !(spacing>0.f) || !(radius>=0.f) ||
           !dimensions[0] || !dimensions[1] || !dimensions[2] ||
           positions.size()>std::numeric_limits<std::uint32_t>::max())
            throw std::invalid_argument("Invalid bricked neighborhood bounds");
        bricked_neighborhood result;
        result.brick_size=brick_size;
        for(int axis=0;axis<3;++axis)
            result.bricks[axis]=
                (dimensions[axis]-1u)/brick_size+1u;
        const std::uint64_t total=
            std::uint64_t(result.bricks[0])*result.bricks[1]*result.bricks[2];
        if(total>std::numeric_limits<std::uint32_t>::max()-1u)
            throw std::length_error("Too many neighborhood bricks");
        result.offsets.reserve(static_cast<std::size_t>(total)+1);
        result.offsets.push_back(0);

        std::vector<bounds> boxes;
        boxes.reserve(positions.size());
        for(const auto& p:positions)boxes.push_back({p,p});
        index tree;
        tree.rebuild(std::move(boxes));
        for(std::uint32_t z=0;z<result.bricks[2];++z)
        for(std::uint32_t y=0;y<result.bricks[1];++y)
        for(std::uint32_t x=0;x<result.bricks[0];++x)
        {
            const std::array<std::uint32_t,3> first{
                x*brick_size,y*brick_size,z*brick_size
            };
            const std::array<std::uint32_t,3> last{
                std::min(dimensions[0]-1u,first[0]+brick_size-1u),
                std::min(dimensions[1]-1u,first[1]+brick_size-1u),
                std::min(dimensions[2]-1u,first[2]+brick_size-1u)
            };
            const vec3 lo{
                origin.x+float(first[0])*spacing-radius,
                origin.y+float(first[1])*spacing-radius,
                origin.z+float(first[2])*spacing-radius
            };
            const vec3 hi{
                origin.x+float(last[0])*spacing+radius,
                origin.y+float(last[1])*spacing+radius,
                origin.z+float(last[2])*spacing+radius
            };
            std::vector<std::uint32_t> found;
            tree.query({lo,hi},[&](std::uint32_t id){found.push_back(id);});
            if(found.size()>max_indices-result.indices.size() ||
               result.indices.size()+found.size()>
                   std::numeric_limits<std::uint32_t>::max())
            {
                result.offsets={std::numeric_limits<std::uint32_t>::max()};
                result.indices.clear();
                return result;
            }
            std::sort(found.begin(),found.end());
            result.indices.insert(result.indices.end(),found.begin(),found.end());
            result.offsets.push_back(
                static_cast<std::uint32_t>(result.indices.size()));
        }
        return result;
    }
}
