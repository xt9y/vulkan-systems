#include <vs/scene/scene.hpp>
#include <iostream>
#include <memory>

// Your project owns one scene. VS algorithms consume it directly.
struct game {
    std::shared_ptr<vs::scene::data> scene=std::make_shared<vs::scene::data>();
    // Add game-only state here (physics, AI, networking, scripting, etc.).
};
int main() {
    game project;
    auto& scene=*project.scene;
    vs::scene::mesh floor{};
    floor.id=1;
    floor.vertices={
        {{-2,0,-2},{0,1,0}},
        {{2,0,-2},{0,1,0}},
        {{0,0,2},{0,1,0}}
    };
    floor.indices={0,2,1};
    scene.meshes.push_back(std::move(floor));
    vs::scene::instance object{};
    object.id=15;
    object.mesh_id=1;
    scene.instances.push_back(object);
    scene.lights.push_back({{0,3,0},5,{1,.8f,.6f},2,true});

    std::size_t triangles=0;
    vs::scene::each_surface(scene,[&](const vs::scene::surface& face) {
        triangles+=face.triangle?1u:2u;
    });
    std::cout<<"instances="<<scene.instances.size()
             <<" renderer triangles="<<triangles
             <<" lights="<<scene.lights.size()<<'\n';
}
