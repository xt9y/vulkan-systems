#pragma once
#include <vk/vk.hpp>
#include <cstdint>
#include <cstddef>
#include <vector>

namespace render
{
    // Non-owning shader selection/binding: nothing is hardcoded to water or PBR.
    // Pipeline layouts/sets remain owned by their creator.
    struct material
    {
        vk::Pipeline pipeline{};
        vk::PipelineLayout layout{};
        std::vector<vk::DescriptorSet> sets;
        std::vector<std::uint32_t> dynamic_offsets;
        std::vector<std::uint8_t> push_constants;
        // Borrowed push-constant memory, avoiding copies per draw; valid until draw ends.
        const void* push_data=nullptr;
        std::size_t push_size=0;
        vk::ShaderStageFlags push_stages{};
    };
}
