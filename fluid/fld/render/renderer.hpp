#pragma once
#include "../core/types.hpp"
#include "../core/system.hpp"
#include <vk/vk.hpp>
#include <string>
#include <vector>

namespace fld::render
{
    struct appearance
    {
        // Linear-light absorption and in-scattering, in inverse metres.
        vec3 absorption{.12f, .045f, .022f};
        vec3 scattering{.005f, .016f, .024f};
        float index_of_refraction = 1.333f;
        float roughness = .035f;
        vec3 light_direction{.45f, .85f, .25f};
    };

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
