#pragma once
#include "editor.hpp"
#include <vk/vk.hpp>
#include <cstdint>
#include <string>
#include <array>

namespace editor::vulkan
{
    // The uploaded x_font.png is embedded in the library as a pixel-exact
    // 16x16 CP437 bitmap atlas (9x16 pixels per character). No external
    // font files, remapping or synthetic fallback are used.
    // Only the Hell2024 menu/background/icon textures need external assets.
    void set_ui_directory(std::string directory);
    std::array<int,128> font_metrics();

    // Decode the embedded atlas and preload menu textures once at startup,
    // before the render thread records any editor commands.
    // record() never accesses the filesystem or decodes images.
    void initialize_font();

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
