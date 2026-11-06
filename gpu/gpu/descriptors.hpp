#pragma once
#include <vk/vk.hpp>
#include <cstdint>
#include <vector>

namespace gpu
{
    // One layout and a pool of equally-shaped sets; arbitrary descriptor types.
    // Caller synchronizes updates against in-flight command buffers.
    class descriptors
    {
    public:
        descriptors()=default;
        ~descriptors(){reset();}
        descriptors(const descriptors&)=delete;
        descriptors& operator=(const descriptors&)=delete;

        void create(std::vector<vk::DescriptorSetLayoutBinding> bindings,
                    std::uint32_t set_count=1);
        void reset();
        vk::DescriptorSetLayout layout() const{return layout_;}
        vk::DescriptorSet set(std::uint32_t index=0) const;
        void write_buffer(std::uint32_t set_index,std::uint32_t binding,
                          vk::DescriptorBufferInfo info,std::uint32_t array_element=0);
        void write_image(std::uint32_t set_index,std::uint32_t binding,
                         vk::DescriptorImageInfo info,std::uint32_t array_element=0);
    private:
        const vk::DescriptorSetLayoutBinding& lookup(
            std::uint32_t binding,std::uint32_t element) const;
        std::vector<vk::DescriptorSetLayoutBinding> bindings_;
        std::vector<vk::DescriptorSet> sets_;
        vk::DescriptorSetLayout layout_{};
        vk::DescriptorPool pool_{};
    };
}
