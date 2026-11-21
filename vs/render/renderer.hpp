#pragma once
#include <cstdint>
#include <vector>
#include <vk/vk.hpp>
#include <vs/render/material.hpp>
#include <vs/render/mesh.hpp>
#include <vs/render/target.hpp>

namespace vs::render
{
// Records into a caller-supplied command buffer. No owned swapchain,
// camera, render targets, or simulation-specific state.
class renderer
{
public:
    explicit renderer(vk::CommandBuffer command);
    void begin_pass(const vk::RenderPassBeginInfo &info);
    void begin_pass(const target &destination, vk::Rect2D area,
                    const std::vector<vk::ClearValue> &clear_values = {});
    void end_pass();
    void viewport(vk::Viewport viewport, vk::Rect2D scissor);
    void draw(const mesh &geometry, const material &surface,
              std::uint32_t instances = 1, std::uint32_t first_instance = 0);
    void draw_indirect(const mesh &geometry, const material &surface,
                       vk::Buffer arguments, vk::DeviceSize offset,
                       std::uint32_t draw_count = 1,
                       std::uint32_t stride = 0);

private:
    void bind(const mesh &geometry, const material &surface) const;
    vk::CommandBuffer command_{};

    bool inside_pass_ = false;
};
} // namespace vs::render
