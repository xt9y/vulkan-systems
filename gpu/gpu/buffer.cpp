#include "buffer.hpp"
#include <stdexcept>

namespace gpu
{
    void buffer::create(vk::DeviceSize bytes,vk::BufferUsageFlags usage,
                        vk::MemoryPropertyFlags memory)
    {
        if(resource_.handle)throw std::logic_error("GPU buffer already created");
        if(!bytes)throw std::invalid_argument("zero-sized GPU buffer");
        resource_=vk::buffer::create(bytes,usage,memory);
    }

    void buffer::reset()
    {
        if(resource_.handle)vk::buffer::destroy(resource_);
        resource_=vk::buffer::resource{};
    }

    void buffer::upload(const void* source,std::size_t bytes,std::size_t offset)
    {
        if(!resource_.handle || !source || offset>resource_.size ||
           bytes>resource_.size-offset)
            throw std::out_of_range("GPU buffer upload range");
        if(!(resource_.properties & vk::MemoryPropertyFlagBits::eHostVisible))
            throw std::logic_error("GPU buffer is not host visible; use staging/compute");
        vk::buffer::write(resource_,source,bytes,offset);
    }

    vk::DescriptorBufferInfo buffer::descriptor(vk::DeviceSize offset,
                                                 vk::DeviceSize range) const
    {
        if(!resource_.handle || offset>=resource_.size ||
           (range!=VK_WHOLE_SIZE && (!range || range>resource_.size-offset)))
            throw std::out_of_range("GPU buffer descriptor range");
        return vk::buffer::descriptor(resource_,offset,range);
    }
}
