#include <vs/bvh/stackless.hpp>
#include <array>
#include <cstdint>
#include <iostream>
#include <vector>

// Build a CPU SAH/stackless triangle BVH, retaining exact triangle geometry
// and repacking triangles into the GPU-friendly leaf order.
int main() {
    std::vector<std::array<float,4>> triangles;
    triangles.reserve(64*64*6);
    const auto vertex=[&](float x,float y,float z) {
        return std::array<float,4>{x,y,z,1.f};
    };
    for(int z=0;z<64;++z)
        for(int x=0;x<64;++x) {
            const float h=float((x*x+z*7)%11)*.035f;
            const float X=float(x),Z=float(z);
            triangles.push_back(vertex(X,h,Z));
            triangles.push_back(vertex(X+1,h,Z));
            triangles.push_back(vertex(X+1,h,Z+1));
            triangles.push_back(vertex(X,h,Z));
            triangles.push_back(vertex(X+1,h,Z+1));
            triangles.push_back(vertex(X,h,Z+1));
        }
    const auto nodes=vs::bvh::build_stackless(triangles);
    std::uint32_t leaves=0,max_leaf=0;
    for(const auto& node:nodes)
        if(const std::uint32_t count=node.first_and_count&15u;count) {
            ++leaves;
            max_leaf=std::max(max_leaf,count);
        }
    std::cout<<"triangles="<<triangles.size()/3
             <<" BVH nodes="<<nodes.size()
             <<" leaves="<<leaves<<" max triangles/leaf="<<max_leaf<<'\n';
}
