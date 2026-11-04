#pragma once
#include "../core/types.hpp"
#include "../core/system.hpp"
#include <vk/vk.hpp>
#include <string>
#include <vector>

namespace fld::render
{
    using appearance=fld::appearance;

    struct input
    {
        vk::ImageView depth{}, scene_color{}, output{};
        vk::Sampler sampler{};
        mat4 view_projection = mat4::identity();
        mat4 inverse_view_projection = mat4::identity();
        std::uint32_t width = 0, height = 0;
        vk::ImageLayout depth_layout = vk::ImageLayout::eDepthStencilReadOnlyOptimal;
        vk::ImageLayout color_layout = vk::ImageLayout::eShaderReadOnlyOptimal;
        bool reversed_depth = false;
        std::uint32_t phase = 0xffffffffu; // all phases by default
        // Phase appearance is the default. Override only for previews
        // or rendering all phases as one material.
        bool use_phase_appearance=true;
        appearance liquid{};
    };

    struct frame
    {
        vk::DescriptorSet descriptors{};
        vk::buffer::resource particles, tile_counts, tile_indices;
    };

    struct state
    {
        vk::DescriptorSetLayout descriptor_layout{};
        vk::DescriptorPool descriptor_pool{};
        vk::PipelineLayout pipeline_layout{};
        vk::Pipeline pipeline{};
        std::vector<frame> frames;
        std::uint32_t width = 0, height = 0;
        std::uint32_t max_particles = 0;
    };

    state *get();
    bool active();
    void init(const std::vector<std::uint32_t> &spirv,
              std::uint32_t frames, std::uint32_t width,
              std::uint32_t height);
    void init(const std::string &spirv_path,
              std::uint32_t frames, std::uint32_t width,
              std::uint32_t height);
    // Call after depth/scene color are readable and output is in GENERAL.
    // The host must wait for the frame fence before reusing a frame index.
    void record(vk::CommandBuffer command, std::uint32_t frame_index,
                const input &target);
    void shutdown();
}
