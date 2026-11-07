#include "trace.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace lumen::trace {
namespace {
constexpr std::uint32_t invalid=UINT32_MAX;

float component(const bvh::vec4& v,int i) {
    return (&v.x)[i];
}

float center(const bvh::triangle& t,int axis) {
    return (component(t.a,axis)+component(t.b,axis)+component(t.c,axis))/3.f;
}
}

mesh build(const std::vector<std::array<float,3>>& vertices) {
    if(vertices.empty() || vertices.size()%3 ||
       vertices.size()/3 > 65536)
        throw std::invalid_argument("Lumen BVH requires 1..65536 triangles");
    mesh result{};
    result.triangles.reserve(vertices.size()/3);
    for(std::size_t i=0;i<vertices.size();i+=3) {
        bvh::triangle t{};
        t.a={vertices[i][0],vertices[i][1],vertices[i][2],0.f};
        t.b={vertices[i+1][0],vertices[i+1][1],vertices[i+1][2],0.f};
        t.c={vertices[i+2][0],vertices[i+2][1],vertices[i+2][2],0.f};
        result.triangles.push_back(t);
    }

    std::vector<std::uint32_t> indices(result.triangles.size());
    std::iota(indices.begin(),indices.end(),0u);
    result.nodes.reserve(indices.size()*2-1);

    const auto build_node=[&](auto&& self,std::size_t first,std::size_t last,
                              std::uint32_t parent)->std::uint32_t {
        const auto index=std::uint32_t(result.nodes.size());
        result.nodes.emplace_back();
        bvh::node& node=result.nodes[index];
        node.links={invalid,invalid,parent,invalid};

        float lo[3]={INFINITY,INFINITY,INFINITY};
        float hi[3]={-INFINITY,-INFINITY,-INFINITY};
        for(std::size_t i=first;i<last;i++) {
            const auto& tri=result.triangles[indices[i]];
            for(int axis=0;axis<3;axis++) {
                const float a=component(tri.a,axis);
                const float b=component(tri.b,axis);
                const float c=component(tri.c,axis);
                lo[axis]=std::min(lo[axis],std::min(a,std::min(b,c)));
                hi[axis]=std::max(hi[axis],std::max(a,std::max(b,c)));
            }
        }
        constexpr float pad=0.0001f;
        node.minimum={lo[0]-pad,lo[1]-pad,lo[2]-pad,0.f};
        node.maximum={hi[0]+pad,hi[1]+pad,hi[2]+pad,0.f};

        if(last-first==1) {
            node.links[3]=indices[first];
            return index;
        }
        int axis=0;
        for(int i=1;i<3;i++)
            if(hi[i]-lo[i]>hi[axis]-lo[axis])axis=i;
        const std::size_t middle=first+(last-first)/2;
        std::nth_element(indices.begin()+first,indices.begin()+middle,
                         indices.begin()+last,
                         [&](std::uint32_t a,std::uint32_t b) {
            return center(result.triangles[a],axis) <
                   center(result.triangles[b],axis);
        });
        const auto left=self(self,first,middle,index);
        const auto right=self(self,middle,last,index);
        result.nodes[index].links[0]=left;
        result.nodes[index].links[1]=right;
        return index;
    };
    build_node(build_node,0,indices.size(),invalid);
    if(result.nodes.size()!=result.triangles.size()*2-1)
        throw std::logic_error("invalid BVH topology");
    return result;
}

} // namespace lumen::trace
