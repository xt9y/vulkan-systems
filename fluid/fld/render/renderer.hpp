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
        std::uint32_t width=0,height=0;
        vk::ImageLayout depth_layout=vk::ImageLayout::eDepthStencilReadOnlyOptimal;
        vk::ImageLayout color_layout=vk::ImageLayout::eGeneral;
        bool reversed_depth=false;
    };

    struct frame
    {
        vk::DescriptorSet descriptors{};
        vk::buffer::resource tile_counts,tile_indices;
    };

    struct state
    {
        vk::DescriptorSetLayout layout{};
        vk::DescriptorPool pool{};
        vk::PipelineLayout pipeline_layout{};
        std::array<vk::Pipeline,3> pipelines{}; // clear, bin, surface
        std::vector<frame> frames;
        std::uint32_t width=0,height=0;
    };

    state *get();
    bool active();
    void init(const std::array<std::string,3> &paths,
              std::uint32_t frames,std::uint32_t width,std::uint32_t height);
    // Composite after opaque lighting; command must have GENERAL output.
    void record(vk::CommandBuffer command,std::uint32_t frame_index,
                const input &target);
    void shutdown();
}
