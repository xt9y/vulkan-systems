#include <array>
#include <cstdint>
#include <memory>
#include <vector>
#include <vs/app/app.hpp>
#include <vs/gpu/image.hpp>
#include <vs/pcloud/pcloud.hpp>

// Full PCloud pipeline: user-supplied triangles/lights, four sampled G-buffer
// images, probes, GI, lighting, tone mapping and swapchain presentation.
// The G-buffer is filled with an analytic flat surface for a standalone demo;
// a game instead fills it with its own rasterizer.
namespace
{
struct frame_inputs
{
    std::array<std::unique_ptr<vs::gpu::image>, 4> images;

    bool filled = false;
};

void initialize(vs::gpu::image &target, vk::Extent2D extent, vk::Format format)
{
    vk::ImageCreateInfo image{};
    image.imageType = vk::ImageType::e2D;
    image.format = format;
    image.extent = vk::Extent3D{extent.width, extent.height, 1};
    image.mipLevels = 1;
    image.arrayLayers = 1;
    image.samples = vk::SampleCountFlagBits::e1;
    image.tiling = vk::ImageTiling::eOptimal;
    image.usage = vk::ImageUsageFlagBits::eSampled |
                  vk::ImageUsageFlagBits::eTransferDst;
    target.create(image, vk::MemoryPropertyFlagBits::eDeviceLocal);
    vk::ImageViewCreateInfo view{};
    view.viewType = vk::ImageViewType::e2D;
    view.format = format;
    view.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
    target.make_view(view);
}

void fill(vk::CommandBuffer cmd, frame_inputs &g)
{
    const std::array<std::array<float, 4>, 4> clear{{
        {{.73f, .61f, .46f, 1}}, // albedo
        {{.5f, 1.f, .5f, 1}},    // encoded +Y normal
        {{.82f, .08f, 1.f, 0}},  // roughness, metallic, occlusion
        {{.96f, 0, 0, 0}}        // depth
    }};

    const vk::ImageSubresourceRange range{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};

    for (std::uint32_t i = 0; i < 4; ++i)
    {
        auto image = g.images[i]->handle();

        vk::ImageMemoryBarrier barrier{};
        barrier.image = image;
        barrier.subresourceRange = range;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.oldLayout = vk::ImageLayout::eUndefined;
        barrier.newLayout = vk::ImageLayout::eTransferDstOptimal;
        barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
                            vk::PipelineStageFlagBits::eTransfer, {}, {}, {}, {barrier});
        vk::ClearColorValue value{};
        value.float32 = clear[i];
        cmd.clearColorImage(image, vk::ImageLayout::eTransferDstOptimal, value, {range});
        barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
        barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
        barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                            vk::PipelineStageFlagBits::eComputeShader, {}, {}, {}, {barrier});
    }

    g.filled = true;
}
} // namespace
int main()
{
    vs::application app({.title = "vs / pcloud - real GI", .width = 960, .height = 540, .frames_in_flight = 1});

    auto scene = std::make_shared<vs::scene::data>();
    scene->lights.push_back({{0, 3, -2}, 7, {1.f, .76f, .5f}, 2.f});
    scene->lights.push_back({{2, 1, 2}, 3, {.3f, .65f, 1.f}, .8f});
    scene->surfaces = {
        {{-5, 0, -5}, {10, 0, 0}, {0, 0, 10}, {0, 1, 0}},
        {{-1, 0, -1}, {0, 2, 0}, {2, 0, 0}, {0, 0, -1}}
    };

    std::vector<std::unique_ptr<frame_inputs>> gbuffers;
    app.on_swapchain([&]
                     {
        vk::device::idle();
        vs::pcloud::release();
        gbuffers.clear(); }, [&]
                     {
        const auto extent=app.extent();
        const auto images=vk::swapchain::get()->images;
        std::vector<vs::pcloud::gbuffer> views;
        for(std::size_t i=0;i<images.size();++i) {
            auto frame=std::make_unique<frame_inputs>();
            for(std::size_t j=0;j<4;++j) {
                frame->images[j]=std::make_unique<vs::gpu::image>();
                const auto format=j==3?vk::Format::eR32Sfloat:
                                       vk::Format::eR8G8B8A8Unorm;
                initialize(*frame->images[j],extent,format);
            }
            views.push_back({frame->images[0]->view(),frame->images[1]->view(),
                             frame->images[2]->view(),frame->images[3]->view()});
            gbuffers.push_back(std::move(frame));
        }
        vs::pcloud::create(views);
        vs::pcloud::set_scene(scene);
        vs::pcloud::synchronize_scene(); });
    app.run([&](vk::CommandBuffer cmd, std::uint32_t image)
            {
        auto& surface=*gbuffers.at(image);
        if(!surface.filled)fill(cmd,surface);
        vs::pcloud::record(cmd,image,{{0,1.5f,5.f},0.f,-.18f,1.0472f}); });
    vk::device::idle();
    vs::pcloud::release();
    gbuffers.clear();
    app.close();
}
