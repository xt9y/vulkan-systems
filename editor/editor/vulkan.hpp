#pragma once
#include "editor.hpp"
#include <vk/vk.hpp>
#include <cstdint>
#include <string>

namespace editor::vulkan
{
    // Optional exact legacy bitmap font from xt9y/hellengine-bak:
    // Rogue/Rogue/res/textures/font/char_1.png ... char_90.png.
    // Set before the first record() call. Missing assets use 5x7 fallback.
    void set_font_directory(std::string directory);

    // Draws opaque unlit pixels directly into an already-open Vulkan color
    // render pass (color attachment index normally zero). A separate overlay
    // pass must LOAD existing color; an in-pass overlay can follow scene draws.
    // No shaders, GPU memory, descriptors, render graph, camera or swapchain
    // are owned by this adapter. Record AFTER the scene, BEFORE endRenderPass.
    // The caller owns command buffer synchronization and framebuffer lifetime.
    void record(vk::CommandBuffer command,
                const draw_list& commands,
                vk::Extent2D extent,
                std::uint32_t color_attachment=0);
}
