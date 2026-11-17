#include <vs/app/app.hpp>
#include <vs/render/renderer.hpp>
#include <vs/render/pipeline.hpp>
#include <chrono>

// A genuine raster pipeline drawing a rotating triangle into the VS swapchain.
// The project (not VS) owns its pipeline layout, shaders, and rendering policy.
int main() {
    vs::application app({.title="vs / render - rotating triangle"});
    vs::render::pipeline triangle;
    vk::PipelineLayout layout{};
    const auto release=[&] {
        triangle.reset();
        if(layout)vk::device::get()->handle.destroyPipelineLayout(layout);
        layout={};
    };
    app.on_swapchain(release,[&] {
        vk::PushConstantRange push{};
        push.stageFlags=vk::ShaderStageFlagBits::eVertex;
        push.offset=0;push.size=sizeof(float);
        vk::PipelineLayoutCreateInfo info{};
        info.pushConstantRangeCount=1;info.pPushConstantRanges=&push;
        layout=vk::device::get()->handle.createPipelineLayout(info);
        vs::render::raster_state state{};
        state.pass=app.color_pass();
        state.layout=layout;
        state.cull=vk::CullModeFlagBits::eNone;
        state.depth_test=state.depth_write=false;
        triangle.create(state,{
            {vk::ShaderStageFlagBits::eVertex,
                "build/shaders/example_triangle.vert.spv"},
            {vk::ShaderStageFlagBits::eFragment,
                "build/shaders/example_triangle.frag.spv"}
        });
    });
    const auto start=std::chrono::steady_clock::now();
    app.run([&](vk::CommandBuffer cmd,std::uint32_t image) {
        const float t=std::chrono::duration<float>(
            std::chrono::steady_clock::now()-start).count();
        app.begin_color(cmd,image,{.025f,.035f,.055f,1.f});
        vs::render::renderer draw(cmd);
        const auto size=app.extent();
        draw.viewport(vk::Viewport{0,0,float(size.width),float(size.height),0,1},
                      vk::Rect2D{{0,0},size});
        vs::render::mesh shape;shape.vertex_count=3;
        vs::render::material material{};
        material.pipeline=triangle.handle();
        material.layout=layout;
        material.push_data=&t;material.push_size=sizeof(t);
        material.push_stages=vk::ShaderStageFlagBits::eVertex;
        draw.draw(shape,material);
        app.end_color(cmd);
    });
    vk::device::idle();
    release();
    app.close();
}
