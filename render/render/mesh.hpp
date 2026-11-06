#pragma once
#include <vk/vk.hpp>
#include <cstdint>
#include <vector>

namespace render
{
    struct vertex_stream
    {
        std::uint32_t binding=0;
        vk::Buffer buffer{};
        vk::DeviceSize offset=0;
    };

    // Non-owning: may point to static, streamed or compute-generated GPU buffers.
    // Pipeline vertex formats are configured independently.
    struct mesh
    {
        std::vector<vertex_stream> streams;
        vk::Buffer index_buffer{};
        vk::DeviceSize index_offset=0;
        vk::IndexType index_type=vk::IndexType::eUint32;
        std::uint32_t vertex_count=0;
        std::uint32_t index_count=0;
        std::uint32_t first_vertex=0;
        std::uint32_t first_index=0;
        std::int32_t vertex_offset=0;
    };
}
