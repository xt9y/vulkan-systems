#pragma once
#include "editor.hpp"
#include <vk/vk.hpp>
#include <cstdint>
#include <string>

namespace editor::vulkan
{
    // Mandatory exact legacy bitmap font from xt9y/hellengine-bak:
    // Rogue/Rogue/res/textures/font/char_1.png ... char_90.png.
    // Configure before initialize_font(); incomplete assets are fatal.
    void set_font_directory(std::string directory);

    // Decode ALL PNGs, compute opaque horizontal glyph spans and cache them.
    // The host MUST call this on its initialization thread, before creating
    // the render thread or recording any editor command buffers.
    // record() never performs disk access, image decoding or font preparation.
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
