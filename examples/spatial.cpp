#include <vs/spatial/index.hpp>
#include <iostream>
#include <vector>

// A user-generated city of 10,000 independent AABBs, queried by a moving
// neighborhood window. The spatial index never owns a game world.
int main() {
    std::vector<vs::spatial::bounds> buildings;
    for(int z=0;z<100;++z)
        for(int x=0;x<100;++x) {
            const vs::spatial::vec3 lo{float(x)*2.f,0,float(z)*2.f};
            buildings.push_back({lo,lo+vs::spatial::vec3{1.f,2.f,1.f}});
        }
    vs::spatial::index tree;
    tree.rebuild(std::move(buildings));
    for(int step=0;step<8;++step) {
        const float x=step*24.f;
        const vs::spatial::bounds camera{
            {x,-1.f,x},{x+16.f,4.f,x+16.f}
        };
        std::size_t visible=0;
        tree.query(camera,[&](std::uint32_t) {++visible;});
        std::cout<<"view "<<step<<": "<<visible<<" intersecting buildings\n";
    }
}
