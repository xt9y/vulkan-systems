#include <vs/app/app.hpp>
#include <vs/pmap/pmap.hpp>
#include <vs/gpu/descriptors.hpp>
#include <vs/render/pipeline.hpp>
#include <vs/render/renderer.hpp>
#include <cstdint>

// Standalone animated progressive photon-mapping renderer. All scene surfaces,
// camera and presentation resources belong to this mini-project.
int main() {
    vs::application app({.title="vs / pmap - Cornell photon GI",
                         .width=960,.height=540,.frames_in_flight=1});
    vs::scene::data scene;
    const auto quad=[](vs::scene::vec3 o,vs::scene::vec3 u,vs::scene::vec3 v,
                       vs::scene::vec3 n,vs::scene::vec3 color) {
        vs::scene::surface result;
        result.origin=o;result.u=u;result.v=v;result.normal=n;
        result.albedo=color;return result;
    };
    scene.surfaces.push_back(quad({-2,0,-3},{0,0,4},{4,0,0},
                                  {0,1,0},{.72f,.68f,.62f}));
    scene.surfaces.push_back(quad({-2,0,-3},{0,4,0},{0,0,4},
                                  {1,0,0},{.85f,.12f,.10f}));
    scene.surfaces.push_back(quad({2,0,-3},{0,0,4},{0,4,0},
                                  {-1,0,0},{.09f,.7f,.18f}));
    scene.surfaces.push_back(quad({-2,0,-3},{4,0,0},{0,4,0},
                                  {0,0,1},{.72f,.73f,.8f}));
    scene.surfaces.push_back(quad({-2,4,-3},{4,0,0},{0,0,4},
                                  {0,-1,0},{.74f,.74f,.74f}));
    auto lamp=quad({-.45f,3.95f,-1.35f},{.9f,0,0},{0,0,.9f},
                   {0,-1,0},{1,1,1});
    lamp.emission={14,12,10};
    scene.emitter=std::uint32_t(scene.surfaces.size());
    scene.surfaces.push_back(lamp);

    vs::pmap::mapper mapper;
    vs::pmap::configuration settings{};
    settings.transport_only=false;
    settings.photons=8192;
    settings.bounces=4;
    settings.denoise_passes=1;

    vs::gpu::descriptors textures;
    vs::render::pipeline present;
    vk::Sampler sampler{};
    vk::PipelineLayout layout{};

    const auto release=[&] {
        present.reset();
        textures.reset();
        if(layout)vk::device::get()->handle.destroyPipelineLayout(layout);
        layout={};
        if(sampler)vk::device::get()->handle.destroySampler(sampler);
        sampler={};
        mapper.destroy();
    };
    app.on_swapchain(release,[&] {
        mapper.create(scene,app.extent(),settings);
        vk::DescriptorSetLayoutBinding binding{};
        binding.binding=0;binding.descriptorCount=1;
        binding.descriptorType=vk::DescriptorType::eCombinedImageSampler;
        binding.stageFlags=vk::ShaderStageFlagBits::eFragment;
        textures.create({binding});
        vk::SamplerCreateInfo info{};
        info.magFilter=info.minFilter=vk::Filter::eLinear;
        info.addressModeU=info.addressModeV=info.addressModeW=
            vk::SamplerAddressMode::eClampToEdge;
        sampler=vk::device::get()->handle.createSampler(info);
        vk::PipelineLayoutCreateInfo pipe_info{};
        const auto set_layout=textures.layout();
        pipe_info.setLayoutCount=1;pipe_info.pSetLayouts=&set_layout;
        layout=vk::device::get()->handle.createPipelineLayout(pipe_info);
        vs::render::raster_state raster{};
        raster.pass=app.color_pass();
        raster.layout=layout;
        raster.cull=vk::CullModeFlagBits::eNone;
        raster.depth_test=raster.depth_write=false;
        present.create(raster,{
            {vk::ShaderStageFlagBits::eVertex,
                "build/shaders/example_fullscreen.vert.spv"},
            {vk::ShaderStageFlagBits::eFragment,
                "build/shaders/example_fullscreen.frag.spv"}
        });
    });

    const vs::scene::camera camera{{0,1.7f,5.f},0.f,0.f,.95f};
    app.run([&](vk::CommandBuffer cmd,std::uint32_t image) {
        const auto output=mapper.record(cmd,camera);
        // Compute writes must become visible to the fullscreen sampler.
        vk::MemoryBarrier barrier{};
        barrier.srcAccessMask=vk::AccessFlagBits::eShaderWrite;
        barrier.dstAccessMask=vk::AccessFlagBits::eShaderRead;
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                            vk::PipelineStageFlagBits::eFragmentShader,
                            {},{barrier},{},{});
        vk::DescriptorImageInfo sampled{};
        sampled.imageView=output.view;
        sampled.sampler=sampler;
        sampled.imageLayout=output.layout;
        textures.write_image(0,0,sampled);

        app.begin_color(cmd,image,{0,0,0,1});
        vs::render::renderer draw(cmd);
        const auto size=app.extent();
        draw.viewport(vk::Viewport{0,0,float(size.width),float(size.height),0,1},
                      vk::Rect2D{{0,0},size});
        vs::render::mesh triangle{};
        triangle.vertex_count=3;
        vs::render::material surface{};
        surface.pipeline=present.handle();surface.layout=layout;
        surface.sets={textures.set()};
        draw.draw(triangle,surface);
        app.end_color(cmd);
    });
    vk::device::idle();
    release();
    app.close();
}
