#pragma once
#include <vk/vk.hpp>
#include <cstdint>

namespace gpu
{
    // Convenience recording. Resources must have appropriate usage flags;
    // synchronization is provided explicitly or through gpu::graph.
    struct transfer
    {
        static void copy(vk::CommandBuffer cmd,vk::Buffer source,
                         vk::Buffer destination,vk::DeviceSize bytes,
                         vk::DeviceSize source_offset=0,
                         vk::DeviceSize destination_offset=0);
        static void fill(vk::CommandBuffer cmd,vk::Buffer destination,
                         vk::DeviceSize offset,vk::DeviceSize bytes,
                         std::uint32_t pattern);
    };
}
