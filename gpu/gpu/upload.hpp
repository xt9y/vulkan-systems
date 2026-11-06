#pragma once

#include <vk/vk.hpp>
#include "transfer.hpp"

#include <algorithm>
#include <cstddef>
#include <deque>
#include <stdexcept>
#include <utility>
#include <vector>

namespace gpu::upload
{
    struct pending {
        vk::CommandBuffer command{};
        vk::Fence fence{};
        std::vector<vk::buffer::resource> staging;
    };

    inline std::deque<pending>& in_flight()
    {
        static std::deque<pending> jobs;
        return jobs;
    }

    inline void retire(bool wait = false)
    {
        auto& jobs = in_flight();
        const auto device = vk::device::get()->handle;
        for (auto it = jobs.begin(); it != jobs.end();)
        {
            const auto result = wait
                ? device.waitForFences(it->fence, true, UINT64_MAX)
                : device.getFenceStatus(it->fence);
            if (result == vk::Result::eNotReady && !wait)
            {
                ++it;
                continue;
            }
            if (result != vk::Result::eSuccess)
            {
                throw std::runtime_error("GPU upload fence failed");
            }

            vk::command::free({it->command});
            for (auto& source : it->staging)
            {
                vk::buffer::destroy(source);
            }
            device.destroyFence(it->fence);
            it = jobs.erase(it);
        }
    }

    inline bool ready()
    {
        retire();
        return in_flight().empty();
    }

    // Buffer copies are enqueued on the graphics queue. Staging memory and
    // command buffers live until their submission fences signal.
    class batch
    {
        vk::CommandBuffer command{};
        std::vector<vk::buffer::resource> staging;
        vk::Fence fence{};
        bool submitted = false;

    public:
        batch() = default;
        batch(const batch&) = delete;
        batch& operator=(const batch&) = delete;

        vk::buffer::resource add(const void* data, std::size_t bytes,
                                 vk::BufferUsageFlags usage,
                                 vk::DeviceSize capacity = 0)
        {
            if (!data || !bytes)
                throw std::invalid_argument("Invalid static mesh upload");

            vk::buffer::resource source{};
            vk::buffer::resource destination{};
            try
            {
                source = vk::buffer::create(
                    bytes, vk::BufferUsageFlagBits::eTransferSrc,
                    vk::MemoryPropertyFlagBits::eHostVisible |
                    vk::MemoryPropertyFlagBits::eHostCoherent
                );
                vk::buffer::write(source, data, bytes);
                destination = vk::buffer::create(
                    std::max<vk::DeviceSize>(bytes, capacity),
                    usage | vk::BufferUsageFlagBits::eTransferDst,
                    vk::MemoryPropertyFlagBits::eDeviceLocal
                );
                if (!command) command = vk::command::begin_one_time();

                gpu::transfer::copy(command, source.handle, destination.handle, bytes);
                staging.push_back(source);
                return destination;
            }
            catch (...)
            {
                if (source.handle) vk::buffer::destroy(source);
                if (destination.handle) vk::buffer::destroy(destination);
                throw;
            }
        }

        // Texture data stays alive in staging until the upload fence signals.
        vk::image::resource add_image(const void* pixels, std::uint32_t width,
                                      std::uint32_t height)
        {
            if (!pixels || !width || !height)
                throw std::invalid_argument("Invalid texture upload");
            const std::size_t bytes = std::size_t(width) * height * 4;
            auto source = vk::buffer::create(
                bytes, vk::BufferUsageFlagBits::eTransferSrc,
                vk::MemoryPropertyFlagBits::eHostVisible |
                vk::MemoryPropertyFlagBits::eHostCoherent
            );
            vk::image::resource image{};
            try
            {
                vk::buffer::write(source, pixels, bytes);
                vk::ImageCreateInfo info{};
                info.imageType = vk::ImageType::e2D;
                info.format = vk::Format::eR8G8B8A8Unorm;
                info.extent = vk::Extent3D{width, height, 1};
                info.mipLevels = 1;
                info.arrayLayers = 1;
                info.samples = vk::SampleCountFlagBits::e1;
                info.tiling = vk::ImageTiling::eOptimal;
                info.usage = vk::ImageUsageFlagBits::eTransferDst |
                    vk::ImageUsageFlagBits::eSampled;
                image = vk::image::create(info, vk::MemoryPropertyFlagBits::eDeviceLocal);
                if (!command) command = vk::command::begin_one_time();

                vk::ImageMemoryBarrier barrier{};
                barrier.image = image.handle;
                barrier.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
                barrier.oldLayout = vk::ImageLayout::eUndefined;
                barrier.newLayout = vk::ImageLayout::eTransferDstOptimal;
                barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;
                command.pipelineBarrier(
                    vk::PipelineStageFlagBits::eTopOfPipe,
                    vk::PipelineStageFlagBits::eTransfer,
                    {}, {}, {}, {barrier}
                );

                vk::BufferImageCopy copy{};
                copy.imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
                copy.imageExtent = vk::Extent3D{width, height, 1};
                command.copyBufferToImage(
                    source.handle, image.handle,
                    vk::ImageLayout::eTransferDstOptimal, {copy}
                );

                barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
                barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;
                barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
                barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
                command.pipelineBarrier(
                    vk::PipelineStageFlagBits::eTransfer,
                    vk::PipelineStageFlagBits::eFragmentShader,
                    {}, {}, {}, {barrier}
                );
                staging.push_back(source);
                return image;
            }
            catch (...)
            {
                if (image.handle) vk::image::destroy(image);
                vk::buffer::destroy(source);
                throw;
            }
        }

        void write(vk::buffer::resource& target, const void* data,
                   std::size_t bytes, std::size_t offset = 0)
        {
            if (!data || !bytes || offset > target.size || bytes > target.size - offset)
                throw std::invalid_argument("Invalid staged buffer update");

            auto source = vk::buffer::create(
                bytes, vk::BufferUsageFlagBits::eTransferSrc,
                vk::MemoryPropertyFlagBits::eHostVisible |
                vk::MemoryPropertyFlagBits::eHostCoherent
            );
            try
            {
                vk::buffer::write(source, data, bytes);
                if (!command) command = vk::command::begin_one_time();
                gpu::transfer::copy(command, source.handle, target.handle, bytes, 0, offset);
                staging.push_back(source);
            }
            catch (...)
            {
                vk::buffer::destroy(source);
                throw;
            }
        }

        void finish()
        {
            if (!command) return;
            const auto device = vk::device::get()->handle;

            // Shader-visible GPU buffers need an explicit transfer-to-read
            // dependency even though upload and rendering use the same queue.
            vk::MemoryBarrier barrier{};
            barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
            barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead |
                vk::AccessFlagBits::eVertexAttributeRead |
                vk::AccessFlagBits::eIndexRead;
            command.pipelineBarrier(
                vk::PipelineStageFlagBits::eTransfer,
                vk::PipelineStageFlagBits::eComputeShader |
                    vk::PipelineStageFlagBits::eVertexShader |
                    vk::PipelineStageFlagBits::eVertexInput,
                {}, {barrier}, {}, {}
            );
            command.end();
            fence = device.createFence({});

            // Allocate the retirement entry before submitting. An allocation
            // failure must not strand already-submitted staging buffers.
            auto& pending = in_flight();
            pending.emplace_back();
            try
            {
                vk::SubmitInfo submit{};
                submit.commandBufferCount = 1;
                submit.pCommandBuffers = &command;
                vk::submit::queue(vk::queue::get()->graphics, {submit}, fence);
                submitted = true;

                auto& entry = pending.back();
                entry.command = command;
                entry.fence = fence;
                entry.staging = std::move(staging);
                command = nullptr;
                fence = nullptr;
                submitted = false;
            }
            catch (...)
            {
                pending.pop_back();
                throw;
            }
        }

        ~batch()
        {
            if (submitted && fence)
            {
                try
                {
                    (void)vk::device::get()->handle.waitForFences(fence, true, UINT64_MAX);
                }
                catch (...)
                {
                }
            }
            if (command) vk::command::free({command});
            for (auto& source : staging)
            {
                vk::buffer::destroy(source);
            }
            if (fence) vk::device::get()->handle.destroyFence(fence);
        }
    };
}
