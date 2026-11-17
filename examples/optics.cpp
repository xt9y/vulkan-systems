#include <vs/app/app.hpp>
#include <vs/pmap/pmap.hpp>
#include <vs/optics/caustics.hpp>
#include <vs/gpu/image.hpp>
#include <vs/gpu/descriptors.hpp>
#include <vs/render/pipeline.hpp>
#include <vs/render/renderer.hpp>
#include <cstdint>

// Glass, mirror and caustic GPU passes with a project-owned scene and depth.
// The sample uses a placeholder depth image; real geometry depth is required
// to see physically correct projected caustics.
int main() {
    vs::application app({.title="vs / optics - glass, mirror and caustics",
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
    auto glass=quad({-.7f,1.f,-1.2f},{0,0,1.1f},{1.4f,0,0},
                    {0,1,0},{.92f,.96f,1.f});
    glass.optical_type=1;glass.optical_strength=.95f;glass.ior=1.52f;
    scene.surfaces.push_back(glass);
    auto mirror=quad({-.9f,.1f,-2.8f},{1.8f,0,0},{0,1.8f,0},
                     {0,0,1},{.9f,.9f,.95f});
    mirror.optical_type=2;mirror.optical_strength=1.f;
    scene.surfaces.push_back(mirror);
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
    vs::gpu::image depth;
    vs::optics::caustics caustics;
    bool prepared_depth=false;
    vs::render::pipeline present;
    vk::Sampler sampler{};
    vk::PipelineLayout layout{};

    const auto release=[&] {
        present.reset();
        caustics.reset();
        depth.reset();
        prepared_depth=false;
        textures.reset();
        if(layout)vk::device::get()->handle.destroyPipelineLayout(layout);
        layout={};
        if(sampler)vk::device::get()->handle.destroySampler(sampler);
        sampler={};
        mapper.destroy();
    };
    app.on_swapchain(release,[&] {
        mapper.create(scene,app.extent(),settings);
        const auto extent=app.extent();
        vk::ImageCreateInfo desc{};
        desc.imageType=vk::ImageType::e2D;
        desc.format=vk::Format::eD32Sfloat;
        desc.extent={extent.width,extent.height,1};
        desc.mipLevels=desc.arrayLayers=1;
        desc.samples=vk::SampleCountFlagBits::e1;
        desc.tiling=vk::ImageTiling::eOptimal;
        desc.usage=vk::ImageUsageFlagBits::eSampled |
                   vk::ImageUsageFlagBits::eTransferDst;
        depth.create(desc,vk::MemoryPropertyFlagBits::eDeviceLocal);
        vk::ImageViewCreateInfo view{};
        view.viewType=vk::ImageViewType::e2D;
        view.format=desc.format;
        view.subresourceRange={vk::ImageAspectFlagBits::eDepth,0,1,0,1};
        depth.make_view(view);
        caustics.create(extent,{depth.view()});
        vk::DescriptorSetLayoutBinding sample{};
        sample.binding=0;sample.descriptorCount=1;
        sample.descriptorType=vk::DescriptorType::eCombinedImageSampler;
        sample.stageFlags=vk::ShaderStageFlagBits::eFragment;
        vk::DescriptorSetLayoutBinding caustic{};
        caustic.binding=1;caustic.descriptorCount=1;
        caustic.descriptorType=vk::DescriptorType::eStorageBuffer;
        caustic.stageFlags=vk::ShaderStageFlagBits::eFragment;
        textures.create({sample,caustic});
        textures.write_buffer(0,1,{caustics.buffer(0),0,VK_WHOLE_SIZE});
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
                "build/shaders/example_optics.frag.spv"}
        });
    });

    const vs::scene::camera camera{{0,1.7f,5.f},0.f,0.f,.95f};
    app.run([&](vk::CommandBuffer cmd,std::uint32_t image) {
        if(!prepared_depth) {
            vk::ImageMemoryBarrier transition{};
            transition.image=depth.handle();
            transition.subresourceRange={vk::ImageAspectFlagBits::eDepth,0,1,0,1};
            transition.srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
            transition.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
            transition.oldLayout=vk::ImageLayout::eUndefined;
            transition.newLayout=vk::ImageLayout::eTransferDstOptimal;
            transition.dstAccessMask=vk::AccessFlagBits::eTransferWrite;
            cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
                vk::PipelineStageFlagBits::eTransfer,{},{},{},{transition});
            const vk::ClearDepthStencilValue value{1.f,0};
            cmd.clearDepthStencilImage(depth.handle(),
                vk::ImageLayout::eTransferDstOptimal,value,{transition.subresourceRange});
            transition.oldLayout=vk::ImageLayout::eTransferDstOptimal;
            transition.newLayout=vk::ImageLayout::eDepthStencilReadOnlyOptimal;
            transition.srcAccessMask=vk::AccessFlagBits::eTransferWrite;
            transition.dstAccessMask=vk::AccessFlagBits::eShaderRead;
            cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                vk::PipelineStageFlagBits::eComputeShader,{},{},{},{transition});
            prepared_depth=true;
        }
        const auto output=mapper.record(cmd,camera);
        vk::MemoryBarrier dependency{};
        dependency.srcAccessMask=vk::AccessFlagBits::eShaderWrite;
        dependency.dstAccessMask=vk::AccessFlagBits::eShaderRead;
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
            vk::PipelineStageFlagBits::eComputeShader,{},{dependency},{},{});
        const auto photons=mapper.photon_buffers();
        caustics.set_photons({photons.sorted,photons.used,photons.capacity});
        const auto size=app.extent();
        const auto view_projection=vk::util::math::perspective(
            camera.vertical_fov,float(size.width)/float(size.height),.1f,100.f) *
            vk::util::math::look_at(camera.position,vk::util::math::vec3{0,1,-1});
        caustics.record(cmd,0,view_projection);
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
