#pragma once
#include <vk/vk.hpp>
#include <vk/util/math/vector.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <vector>

namespace bvh
{
    // Two tightly packed 16-byte groups, identical in C++ and GLSL std430:
    // (vec3 minimum, uint next), (vec3 maximum, uint first_and_count).
    // The lower four bits store a 0..8 triangle leaf count; the remaining
    // bits store its first triangle index. Exact bounds remain 32-bit floats.
    struct alignas(16) stackless_node
    {
        std::array<float,3> minimum{};
        std::uint32_t next=0;
        std::array<float,3> maximum{};
        std::uint32_t first_and_count=0;
    };
    static_assert(sizeof(stackless_node)==32);
    static_assert(offsetof(stackless_node,next)==12);
    static_assert(offsetof(stackless_node,maximum)==16);
    static_assert(offsetof(stackless_node,first_and_count)==28);

    // Input/output triangles are three consecutive vec4 vertices per face.
    // Reorders the triangles into leaf order using binned surface-area
    // heuristic splits. Triangle positions and exact any-hit tests are
    // unchanged; this only reduces the average number of BVH visits.
    inline std::vector<stackless_node> build_stackless(
        std::vector<std::array<float,4>>& triangles)
    {
        using namespace vk::util::math;
        struct primitive
        {
            std::array<vec3,3> positions;
            vec3 lo{},hi{},center{};
        };
        std::vector<primitive> primitives;
        primitives.reserve(triangles.size()/3);
        for(std::size_t i=0;i+2<triangles.size();i+=3)
        {
            const vec3 a{triangles[i][0],triangles[i][1],triangles[i][2]};
            const vec3 b{triangles[i+1][0],triangles[i+1][1],triangles[i+1][2]};
            const vec3 c{triangles[i+2][0],triangles[i+2][1],triangles[i+2][2]};
            const vec3 lo{std::min({a.x,b.x,c.x}),std::min({a.y,b.y,c.y}),
                          std::min({a.z,b.z,c.z})};
            const vec3 hi{std::max({a.x,b.x,c.x}),std::max({a.y,b.y,c.y}),
                          std::max({a.z,b.z,c.z})};
            primitives.push_back({{a,b,c},lo,hi,(lo+hi)*0.5f});
        }
        std::vector<std::uint32_t> order(primitives.size());
        std::iota(order.begin(),order.end(),0u);
        std::vector<stackless_node> nodes;
        std::vector<std::array<float,4>> ordered;
        ordered.reserve(triangles.size());
        const auto axis=[](vec3 value,int i)->float {
            return i==0?value.x:(i==1?value.y:value.z);
        };
        std::function<void(std::size_t,std::size_t)> build=
            [&](std::size_t first,std::size_t end)
        {
            const auto infinity=std::numeric_limits<float>::infinity();
            vec3 lo{infinity,infinity,infinity},hi{-infinity,-infinity,-infinity};
            for(std::size_t i=first;i<end;++i)
            {
                const auto& p=primitives[order[i]];
                lo={std::min(lo.x,p.lo.x),std::min(lo.y,p.lo.y),std::min(lo.z,p.lo.z)};
                hi={std::max(hi.x,p.hi.x),std::max(hi.y,p.hi.y),std::max(hi.z,p.hi.z)};
            }
            stackless_node node{};
            node.minimum={lo.x-0.0001f,lo.y-0.0001f,lo.z-0.0001f};
            node.maximum={hi.x+0.0001f,hi.y+0.0001f,hi.z+0.0001f};
            const auto node_index=nodes.size();
            nodes.push_back(node);
            constexpr std::size_t minimum_leaf=4;
            constexpr std::size_t maximum_leaf=8;
            constexpr int bin_count=12;
            std::size_t middle=first;
            if(end-first>minimum_leaf)
            {
                const float inf=std::numeric_limits<float>::infinity();
                const auto area=[](vec3 low,vec3 high)
                {
                    const auto extent=high-low;
                    return 2.0f*(extent.x*extent.y+
                                 extent.y*extent.z+extent.z*extent.x);
                };
                struct bin
                {
                    vec3 lo,hi;
                    std::uint32_t count=0;
                    bin()
                        :lo{std::numeric_limits<float>::infinity(),
                            std::numeric_limits<float>::infinity(),
                            std::numeric_limits<float>::infinity()},
                         hi{-std::numeric_limits<float>::infinity(),
                            -std::numeric_limits<float>::infinity(),
                            -std::numeric_limits<float>::infinity()}{}
                    void add(const primitive& p)
                    {
                        lo={std::min(lo.x,p.lo.x),std::min(lo.y,p.lo.y),
                            std::min(lo.z,p.lo.z)};
                        hi={std::max(hi.x,p.hi.x),std::max(hi.y,p.hi.y),
                            std::max(hi.z,p.hi.z)};
                        ++count;
                    }
                    void merge(const bin& other)
                    {
                        if(!other.count)return;
                        lo={std::min(lo.x,other.lo.x),
                            std::min(lo.y,other.lo.y),
                            std::min(lo.z,other.lo.z)};
                        hi={std::max(hi.x,other.hi.x),
                            std::max(hi.y,other.hi.y),
                            std::max(hi.z,other.hi.z)};
                        count+=other.count;
                    }
                };
                vec3 center_lo{inf,inf,inf};
                vec3 center_hi{-inf,-inf,-inf};
                for(std::size_t i=first;i<end;++i)
                {
                    const auto& center=primitives[order[i]].center;
                    center_lo={std::min(center_lo.x,center.x),
                               std::min(center_lo.y,center.y),
                               std::min(center_lo.z,center.z)};
                    center_hi={std::max(center_hi.x,center.x),
                               std::max(center_hi.y,center.y),
                               std::max(center_hi.z,center.z)};
                }
                const auto bin_index=[&](std::uint32_t primitive_index,
                                         int direction)
                {
                    const float low=axis(center_lo,direction);
                    const float span=axis(center_hi,direction)-low;
                    if(!(span>0.0f))return 0;
                    const float fraction=
                        (axis(primitives[primitive_index].center,direction)-low)/
                        span;
                    return std::clamp(int(fraction*float(bin_count)),0,bin_count-1);
                };
                const float parent_area=area(lo,hi);
                float best_cost=std::numeric_limits<float>::infinity();
                int best_axis=-1,best_bin=0;
                if(std::isfinite(parent_area) && parent_area>0.0f)
                for(int direction=0;direction<3;++direction)
                {
                    if(!(axis(center_hi,direction)>axis(center_lo,direction)))
                        continue;
                    std::array<bin,bin_count> bins{};
                    for(std::size_t i=first;i<end;++i)
                        bins[bin_index(order[i],direction)].add(primitives[order[i]]);
                    std::array<bin,bin_count> left{},right{};
                    bin prefix,suffix;
                    for(int i=0;i<bin_count;++i)
                    {
                        prefix.merge(bins[i]);
                        left[i]=prefix;
                    }
                    for(int i=bin_count-1;i>=0;--i)
                    {
                        suffix.merge(bins[i]);
                        right[i]=suffix;
                    }
                    for(int i=0;i<bin_count-1;++i)
                    {
                        if(!left[i].count || !right[i+1].count)continue;
                        const float cost=area(left[i].lo,left[i].hi)*
                                         float(left[i].count)+
                                         area(right[i+1].lo,right[i+1].hi)*
                                         float(right[i+1].count);
                        if(cost<best_cost)
                        {
                            best_cost=cost;
                            best_axis=direction;
                            best_bin=i;
                        }
                    }
                }
                // One node test is cheaper than several ray/triangle tests.
                // Prevent excessive fragmentation where the surfaces overlap.
                if(best_axis>=0 &&
                   best_cost+2.0f*parent_area<
                       parent_area*float(end-first))
                {
                    auto position=std::stable_partition(
                        order.begin()+first,order.begin()+end,
                        [&](std::uint32_t id)
                        {
                            return bin_index(id,best_axis)<=best_bin;
                        });
                    middle=std::size_t(position-order.begin());
                }
                // Large, degenerate regions still need bounded leaf sizes.
                if((middle<=first || middle>=end) &&
                   end-first>maximum_leaf)
                {
                    const vec3 span=hi-lo;
                    const int split=span.x>=span.y && span.x>=span.z ? 0
                        : span.y>=span.z ? 1 : 2;
                    middle=first+(end-first)/2;
                    std::nth_element(order.begin()+first,order.begin()+middle,
                        order.begin()+end,[&](std::uint32_t a,std::uint32_t b)
                        {
                            return axis(primitives[a].center,split)<
                                   axis(primitives[b].center,split);
                        });
                }
            }
            if(middle<=first || middle>=end)
            {
                const auto first_triangle=ordered.size()/3;
                if(first_triangle>(std::numeric_limits<std::uint32_t>::max()>>4))
                    throw std::length_error("Shadow BVH exceeds packed triangle range");
                nodes[node_index].first_and_count=
                    (static_cast<std::uint32_t>(first_triangle)<<4) |
                    static_cast<std::uint32_t>(end-first);
                for(std::size_t i=first;i<end;++i)
                    for(const auto& p:primitives[order[i]].positions)
                        ordered.push_back({p.x,p.y,p.z,0.0f});
            }
            else
            {
                build(first,middle);
                build(middle,end);
            }
            nodes[node_index].next=static_cast<std::uint32_t>(nodes.size());
        };
        if(!order.empty())build(0,order.size());
        triangles=std::move(ordered);
        return nodes;
    }
}
