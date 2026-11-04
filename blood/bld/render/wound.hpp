#pragma once
#include "../core/types.hpp"
#include <vk/vk.hpp>
#include <unordered_map>

namespace bld::render::wound
{
    struct attachment
    {
        std::vector<vk::buffer::resource> buffers;
        std::vector<std::uint64_t> generation;
    };

    struct state
    {
        std::uint32_t frames = 0;
        vk::buffer::resource fallback;
        std::unordered_map<instance_id, attachment> instances;
        std::unordered_map<std::uint32_t, attachment> batches;
    };

    state *get();
    void init(std::uint32_t frames_in_flight);
    // Bind this SSBO as the wound buffer for a skinned actor.
    vk::DescriptorBufferInfo descriptor(instance_id id, std::uint32_t frame_index);
    // Instance order equals the skinned draw's instance order. Shader indexes
    // the packed SSBO with its instance index and the atlas resolution.
    vk::DescriptorBufferInfo descriptor_batch(
        std::uint32_t mesh_id, const std::vector<instance_id> &instances,
        std::uint32_t frame_index);
    bool has_wounds(const std::vector<instance_id> &instances);
    void clear_batches();
    void remove(instance_id id);
    void shutdown();
}
