#pragma once
#include <vk/vk.hpp>
#include <cstddef>

namespace gpu
{
    // Owned VulkanEngine buffer. Lifetime must exceed pending GPU submissions.
    class buffer
    {
    public:
        buffer()=default;
        ~buffer(){reset();}
        buffer(const buffer&)=delete;
        buffer& operator=(const buffer&)=delete;

        void create(vk::DeviceSize bytes,vk::BufferUsageFlags usage,
                    vk::MemoryPropertyFlags memory);
        void reset();
        void upload(const void* data,std::size_t bytes,std::size_t offset=0);
        vk::Buffer handle() const {return resource_.handle;}
        vk::DeviceSize size() const {return resource_.size;}
        vk::DescriptorBufferInfo descriptor(vk::DeviceSize offset=0,
                                             vk::DeviceSize range=VK_WHOLE_SIZE) const;
        vk::buffer::resource& resource(){return resource_;}
        const vk::buffer::resource& resource() const{return resource_;}
    private:
        vk::buffer::resource resource_{};
    };
}
