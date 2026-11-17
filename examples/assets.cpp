#include <vs/app/app.hpp>
#include <vs/assets/model.hpp>
#include <vs/assets/animation.hpp>
#include <vs/assets/gpu_assets.hpp>
#include <chrono>
#include <iostream>

// Drag in any .glb/.gltf: import PBR meshes + textures, upload them to the GPU,
// and continuously evaluate animation joints. The caller owns the model data.
int main(int argc,char** argv) {
    if(argc!=2) {
        std::cerr<<"usage: c build run vs-assets path/to/model.glb\n";
        return 2;
    }
    vs::scene::data imported=vs::model::load(argv[1]);
    std::cout<<"meshes="<<imported.meshes.size()
             <<" instances="<<imported.instances.size()
             <<" textures="<<imported.textures.size()
             <<" skins="<<imported.skins.size()<<'\n';
    vs::application app({.title="vs / assets - glTF and Ozz"});
    vs::gpu_assets gpu;
    gpu.upload(imported.meshes,imported.textures);
    vs::animation::synchronize(imported.skins);
    const auto start=std::chrono::steady_clock::now();
    std::vector<vs::animation::matrix> palette;
    app.run([&](vk::CommandBuffer cmd,std::uint32_t image) {
        const float seconds=std::chrono::duration<float>(
            std::chrono::steady_clock::now()-start).count();
        if(!imported.skins.empty()&&!imported.skins[0].clips.empty())
            vs::animation::sample(imported.skins[0].id,0,seconds,palette);
        app.clear(cmd,image,{.06f,.09f,.12f,1.f});
    });
    vk::device::idle();
    gpu.reset();
    vs::animation::reset();
    app.close();
}
