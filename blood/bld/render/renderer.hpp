#pragma once
#include "../core/types.hpp"
#include <vk/vk.hpp>

namespace bld::render
{
    struct input
    {
        std::uint32_t width = 0, height = 0;
        mat4 view_projection = mat4::identity();
        mat4 inverse_view_projection = mat4::identity();
        vk::ImageView depth{}, normals{}, output{};
        vk::Sampler sampler{};
        vk::ImageLayout depth_layout = vk::ImageLayout::eDepthStencilReadOnlyOptimal;
        vk::ImageLayout normal_layout = vk::ImageLayout::eShaderReadOnlyOptimal;
        bool reverse_depth = false;
        bool encoded_normals = true;
    };

    struct frame
    {
        vk::DescriptorSet set{};
        vk::buffer::resource decals, decal_counts, decal_indices;
        vk::buffer::resource drops, drop_counts, drop_indices;
    };

    struct state
    {
        vk::DescriptorSetLayout descriptor_layout{};
        vk::DescriptorPool descriptor_pool{};
        vk::PipelineLayout layout{};
        vk::Pipeline pipeline{};
        std::vector<frame> frames;
        std::uint32_t max_width = 0, max_height = 0;
        std::uint32_t max_decals = 1024, max_drops = 512;
    };

    state *get();
    // Shader binaries are supplied by the application: no compiler at runtime.
    void init(const std::vector<std::uint32_t> &shader_spirv,
              std::uint32_t frames_in_flight,
              std::uint32_t max_width, std::uint32_t max_height);
    // Call in the host's render graph after GBuffer, before lighting/compositing.
    // The host transitions output to GENERAL and inputs to readable layouts.
    // Wait for frame fences before reusing frame_index or releasing GPU resources.
    void record(vk::CommandBuffer command, std::uint32_t frame_index,
                const input &targets);
    void shutdown();
}
