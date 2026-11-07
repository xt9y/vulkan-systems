#pragma once
#include "editor.hpp"
#include <vk/vk.hpp>
#include <cstdint>

namespace editor::vulkan
{
    // Draws opaque unlit pixels directly into an already-open Vulkan color
    // render pass (attachment LOAD, color attachment index normally zero).
    // No shaders, GPU memory, descriptors, render graph, camera or swapchain
    // are owned by this adapter. Record AFTER the scene, BEFORE endRenderPass.
    // The caller owns command buffer synchronization and framebuffer lifetime.
    void record(vk::CommandBuffer command,
                const draw_list& commands,
                vk::Extent2D extent,
                std::uint32_t color_attachment=0);
}
