#pragma once
#include <vk/vk.hpp>
#include <stdexcept>

namespace gpu
{
    // VulkanWrapper owns the logical device and queues. No second device
    // initialization and no hidden renderer/swapchain dependencies.
    struct device
    {
        static vk::Device handle()
        {
            const auto value=vk::device::get()->handle;
            if(!value)throw std::logic_error("VulkanWrapper device not initialized");
            return value;
        }
        static vk::Queue graphics(){return vk::queue::get()->graphics;}
        static vk::Queue compute(){return vk::queue::get()->compute;}
        static vk::Queue transfer(){return vk::queue::get()->transfer;}
    };
}
