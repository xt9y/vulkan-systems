#pragma once
#include <vk/vk.hpp>
#include <vk/util/math/vector.hpp>
#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <vector>

namespace bvh
{
    // Stackless, preorder any-hit BVH for shader paths requiring a compact,
    // directly readable tree. The node layout matches three vec4/uvec4 in
    // std430; info = (subtree_end, first_triangle, triangle_count, 0).
    struct alignas(16) stackless_node
    {
        std::array<float,4> minimum{};
        std::array<float,4> maximum{};
        std::array<std::uint32_t,4> info{};
    };
    static_assert(sizeof(stackless_node)==48);

    // Input/output triangles are three consecutive vec4 vertices per face.
    // Reorders the triangles into leaf order, preserving the existing
    // median-split traversal and the exact vertex positions.
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
            node.minimum={lo.x-0.0001f,lo.y-0.0001f,lo.z-0.0001f,0.0f};
            node.maximum={hi.x+0.0001f,hi.y+0.0001f,hi.z+0.0001f,0.0f};
            const auto node_index=nodes.size();
            nodes.push_back(node);
            if(end-first<=8)
            {
                nodes[node_index].info[1]=static_cast<std::uint32_t>(ordered.size()/3);
                nodes[node_index].info[2]=static_cast<std::uint32_t>(end-first);
                for(std::size_t i=first;i<end;++i)
                    for(const auto& p:primitives[order[i]].positions)
                        ordered.push_back({p.x,p.y,p.z,0.0f});
            }
            else
            {
                const vec3 span=hi-lo;
                const int split=span.x>=span.y && span.x>=span.z ? 0
                    : span.y>=span.z ? 1 : 2;
                const auto middle=first+(end-first)/2;
                std::nth_element(order.begin()+first,order.begin()+middle,
                    order.begin()+end,[&](std::uint32_t a,std::uint32_t b)
                    {
                        return axis(primitives[a].center,split)<
                               axis(primitives[b].center,split);
                    });
                build(first,middle);
                build(middle,end);
            }
            nodes[node_index].info[0]=static_cast<std::uint32_t>(nodes.size());
        };
        if(!order.empty())build(0,order.size());
        triangles=std::move(ordered);
        return nodes;
    }
}
