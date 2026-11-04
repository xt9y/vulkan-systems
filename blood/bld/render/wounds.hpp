#pragma once
#include "../core/types.hpp"
#include <vk/vk.hpp>
#include <unordered_map>
#include <vector>

namespace bld::render::wounds
{
    struct cache
    {
        std::vector<vk::buffer::resource> buffers;
        std::vector<std::uint64_t> generations;
    };
    struct state
    {
        std::uint32_t frames=0;
        vk::buffer::resource fallback;
        std::unordered_map<instance_id,cache> actors;
        std::unordered_map<std::uint32_t,cache> batches;
    };
    state *get();
    void init(std::uint32_t frames);
    bool visible(instance_id id);
    vk::DescriptorBufferInfo descriptor(instance_id id,std::uint32_t frame);
    vk::DescriptorBufferInfo descriptor_batch(
        std::uint32_t mesh_id,const std::vector<instance_id> &instances,
        std::uint32_t frame);
    void clear_batches();
    void shutdown();
}
