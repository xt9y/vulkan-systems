#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <vs/gpu/compute.hpp>
#include <vs/gpu/descriptors.hpp>
#include <vs/gpu/image.hpp>

namespace vs::gpu
{
// Reusable hierarchical depth buffer. Reduces forward-Z depth by maximum
// (reverse-Z by minimum) and exposes all mip levels for future occlusion.
// The source depth image remains owned by the renderer.
class depth_pyramid
{
public:
    depth_pyramid() = default;
    ~depth_pyramid() { reset(); }
    depth_pyramid(const depth_pyramid &) = delete;
    depth_pyramid &operator=(const depth_pyramid &) = delete;

    void create(vk::ImageView depth, vk::Extent2D source_size,
                vk::ImageLayout depth_layout = vk::ImageLayout::eDepthStencilReadOnlyOptimal,
                const std::string &shader = "build/shaders/gpu_hiz.comp.spv",
                bool reverse_z = false);
    // Establish the descriptor layout before binding the pyramid.
    // Depth values become usable after the first build.
    void prepare(vk::CommandBuffer command);
    // Caller ensures the source image is shader-readable and this pyramid
    // has no in-flight readers. Use one pyramid per frame-in-flight.
    void build(vk::CommandBuffer command);
    void reset() noexcept;
    vk::ImageView view() const noexcept { return image_.view(); }
    vk::Sampler sampler() const noexcept { return sampler_; }
    std::uint32_t levels() const noexcept { return levels_; }
    vk::Extent2D extent() const noexcept { return extent_; }

private:
    image image_;
    compute kernel_;
    descriptors descriptors_;

    std::vector<vk::ImageView> level_views_;
    vk::Sampler sampler_{};
    vk::Extent2D extent_{};
    std::uint32_t levels_ = 0, reverse_ = 0;

    bool initialized_ = false;
};
} // namespace vs::gpu
