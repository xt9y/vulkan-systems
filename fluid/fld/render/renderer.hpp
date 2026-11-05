#pragma once
#include "../core/types.hpp"
#include <vk/vk.hpp>
#include <array>
#include <string>
#include <vector>

namespace fld::render
{
    struct input
    {
        vk::ImageView depth{},scene_color{},output{};
        vk::Sampler sampler{};
        mat4 view_projection=mat4::identity();
        std::uint32_t width=0,height=0,low_width=0,low_height=0;
        vk::ImageLayout depth_layout=vk::ImageLayout::eDepthStencilReadOnlyOptimal;
        vk::ImageLayout color_layout=vk::ImageLayout::eGeneral;
        bool reversed_depth=false;
        // Optional debug overlay, linear RGB; zero opacity restores the
        // physically based transparent appearance without changing physics.
        vec3 debug_color{0.f,1.f,0.f};
        float debug_opacity=0.f;
    };

    struct frame
    {
        vk::DescriptorSet descriptors{};
        vk::buffer::resource tile_counts,tile_indices;
        vk::image::resource low_image{};
        vk::ImageView low_view{};
    };

    struct state
    {
        vk::DescriptorSetLayout layout{};
        vk::DescriptorPool pool{};
        vk::PipelineLayout pipeline_layout{};
        std::array<vk::Pipeline,4> pipelines{}; // clear, bin, surface, upscale
        std::vector<frame> frames;
        std::uint32_t width=0,height=0;
    };

    state *get();
    bool active();
    void init(const std::array<std::string,4> &paths,
              std::uint32_t frames,std::uint32_t width,std::uint32_t height,
              float render_scale=.5f);
    // Composite after opaque lighting; command must have GENERAL output.
    void record(vk::CommandBuffer command,std::uint32_t frame_index,
                const input &target);
    void shutdown();
}
