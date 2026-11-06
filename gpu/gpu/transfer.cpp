#include "transfer.hpp"
#include <stdexcept>

namespace gpu
{
    void transfer::copy(vk::CommandBuffer cmd,vk::Buffer source,
                        vk::Buffer destination,vk::DeviceSize bytes,
                        vk::DeviceSize source_offset,vk::DeviceSize destination_offset)
    {
        if(!cmd || !source || !destination || !bytes)
            throw std::invalid_argument("invalid GPU buffer copy");
        vk::BufferCopy range{};
        range.srcOffset=source_offset;
        range.dstOffset=destination_offset;
        range.size=bytes;
        vk::record::copy(cmd,source,destination,range);
    }

    void transfer::fill(vk::CommandBuffer cmd,vk::Buffer destination,
                        vk::DeviceSize offset,vk::DeviceSize bytes,
                        std::uint32_t pattern)
    {
        if(!cmd || !destination || (offset&3u) ||
           (bytes!=VK_WHOLE_SIZE && (!bytes || (bytes&3u))))
            throw std::invalid_argument("invalid GPU buffer fill");
        cmd.fillBuffer(destination,offset,bytes,pattern);
    }
}
