#include <vs/app/app.hpp>
#include <vs/lumen/lumen.hpp>
#include <cstdint>

// Standalone Cornell-box GI: cards, probes, screen radiance and reflections.
// No framework-owned world; the renderer receives an explicit demo setting.
int main() {
    vs::application app({.title="vs / lumen - Cornell GI",.frames_in_flight=1});
    // Same caller-owned scene format used by PMap and PCloud.
    vs::scene::data room;
    const auto quad=[](vs::scene::vec3 a,vs::scene::vec3 u,
                       vs::scene::vec3 v,vs::scene::vec3 normal,
                       vs::scene::vec3 albedo) {
        vs::scene::surface face{};
        face.origin=a;face.u=u;face.v=v;face.normal=normal;
        face.albedo=albedo;
        return face;
    };
    room.surfaces={
        quad({-2,0,-3},{4,0,0},{0,0,4},{0,1,0},{.75f,.74f,.7f}),
        quad({-2,0,-3},{0,4,0},{0,0,4},{1,0,0},{.8f,.1f,.12f}),
        quad({2,0,-3},{0,0,4},{0,4,0},{-1,0,0},{.1f,.7f,.15f}),
        quad({-2,0,-3},{4,0,0},{0,4,0},{0,0,1},{.7f,.7f,.7f}),
        quad({-2,4,-3},{0,0,4},{4,0,0},{0,-1,0},{.8f,.8f,.8f})
    };
    room.lights.push_back({{0,3.8f,-1},3,{1,.9f,.7f},8,true});
    vs::lumen::system gi;
    vs::lumen::settings settings{};
    settings.demo=vs::lumen::demo_scene::none;
    settings.probe_updates_per_frame=64;
    settings.rays_per_probe=8;
    app.on_swapchain([&]{gi.reset();},[&]{
        gi.create(app.color_pass(),app.format(),app.extent(),settings,room);
    });
    std::uint64_t frame=0;
    app.run([&](vk::CommandBuffer cmd,std::uint32_t image) {
        vs::lumen::frame current{};
        current.index=frame++;
        current.delta_seconds=1.f/60.f;
        current.view.position={0,0,3.2f};
        current.view.vertical_fov=39.f*3.14159265358979323846f/180.f;
        gi.record(cmd,current);
        app.begin_color(cmd,image,{0,0,0,1});
        gi.compose(cmd);
        app.end_color(cmd);
    });
    vk::device::idle();
    gi.reset();
    app.close();
}
