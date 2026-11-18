#pragma once
#include <vk/vk.hpp>
#include <vs/gpu/transfer.hpp>
#include <algorithm>
#include <cstddef>
#include <deque>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace vs::gpu::upload
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
        std::size_t shared_staging = std::numeric_limits<std::size_t>::max();
        vk::DeviceSize shared_used = 0;
        static constexpr vk::DeviceSize staging_block = 2u * 1024u * 1024u;

        struct staged_range {
            vk::Buffer buffer{};
            vk::DeviceSize offset = 0;
        };

        // One persistently mapped, host-coherent block backs many small mesh,
        // index, page and texture uploads. Offsets are kept distinct until
        // the transfer fence retires the complete batch.
        staged_range stage(const void* data, std::size_t bytes)
        {
            constexpr vk::DeviceSize alignment = 16;
            const bool dedicated = bytes > staging_block;
            vk::DeviceSize offset = (shared_used + alignment - 1) &
                                    ~(alignment - 1);
            if(dedicated || shared_staging == std::numeric_limits<std::size_t>::max() ||
               offset > staging_block || bytes > staging_block - offset) {
                auto resource = vk::buffer::create(
                    dedicated ? bytes : staging_block,
                    vk::BufferUsageFlagBits::eTransferSrc,
                    vk::MemoryPropertyFlagBits::eHostVisible |
                    vk::MemoryPropertyFlagBits::eHostCoherent);
                try {
                    staging.push_back(resource);
                } catch (...) {
                    vk::buffer::destroy(resource);
                    throw;
                }
                if(!dedicated) {
                    shared_staging = staging.size() - 1u;
                    shared_used = 0;
                }
                offset = 0;
            }
            const std::size_t index = dedicated ? staging.size() - 1u :
                shared_staging;
            auto& source = staging[index];
            vk::buffer::write(source, data, bytes, offset);
            if(!dedicated) shared_used = offset + bytes;
            return {source.handle,offset};
        }

    public:
        batch() = default;
        batch(const batch&) = delete;
        batch& operator=(const batch&) = delete;

        vk::buffer::resource add(const void* data, std::size_t bytes,
                                 vk::BufferUsageFlags usage,
                                 vk::DeviceSize capacity = 0)
        {
            if (!data || !bytes)
                throw std::invalid_argument("Invalid GPU buffer upload");

            vk::buffer::resource destination{};
            const auto source = stage(data, bytes);
            try
            {
                destination = vk::buffer::create(
                    std::max<vk::DeviceSize>(bytes, capacity),
                    usage | vk::BufferUsageFlagBits::eTransferDst,
                    vk::MemoryPropertyFlagBits::eDeviceLocal
                );
                if (!command) command = vk::command::begin_one_time();
                ::vs::gpu::transfer::copy(command, source.buffer,
                                           destination.handle, bytes,
                                           source.offset);
                return destination;
            }
            catch (...)
            {
                if (destination.handle) vk::buffer::destroy(destination);
                throw;
            }
        }

        // Generate the complete mip chain on the upload queue. If the device
        // cannot linearly blit RGBA8 images, upload a single level safely.
        // Staging buffers remain alive until the submission fence signals.
        vk::image::resource add_image(const void* pixels, std::uint32_t width,
                                      std::uint32_t height,
                                      std::uint32_t* uploaded_levels = nullptr)
        {
            if (!pixels || !width || !height)
                throw std::invalid_argument("Invalid texture upload");
            if (std::size_t(width) > std::numeric_limits<std::size_t>::max() /
                                     std::size_t(height) / 4u)
                throw std::overflow_error("RGBA8 texture size overflow");
            const std::size_t bytes = std::size_t(width) * height * 4u;
            const auto properties = vk::physical_device::get()->handle
                .getFormatProperties(vk::Format::eR8G8B8A8Unorm);
            const auto required = vk::FormatFeatureFlagBits::eBlitSrc |
                                  vk::FormatFeatureFlagBits::eBlitDst |
                                  vk::FormatFeatureFlagBits::eSampledImageFilterLinear;
            std::uint32_t levels = 1;
            if ((properties.optimalTilingFeatures & required) == required) {
                auto w=width, h=height;
                while (w>1 || h>1) {
                    w=std::max(1u,w/2u);
                    h=std::max(1u,h/2u);
                    ++levels;
                }
            }
            const auto source = stage(pixels,bytes);
            vk::image::resource image{};
            try
            {
                vk::ImageCreateInfo info{};
                info.imageType=vk::ImageType::e2D;
                info.format=vk::Format::eR8G8B8A8Unorm;
                info.extent=vk::Extent3D{width,height,1u};
                info.mipLevels=levels;
                info.arrayLayers=1;
                info.samples=vk::SampleCountFlagBits::e1;
                info.tiling=vk::ImageTiling::eOptimal;
                info.usage=vk::ImageUsageFlagBits::eTransferDst |
                           vk::ImageUsageFlagBits::eSampled;
                if(levels>1)info.usage|=vk::ImageUsageFlagBits::eTransferSrc;
                image=vk::image::create(info,vk::MemoryPropertyFlagBits::eDeviceLocal);
                if (!command)command=vk::command::begin_one_time();

                vk::ImageMemoryBarrier barrier{};
                barrier.image=image.handle;
                barrier.subresourceRange=vk::ImageSubresourceRange{
                    vk::ImageAspectFlagBits::eColor,0,levels,0,1};
                barrier.oldLayout=vk::ImageLayout::eUndefined;
                barrier.newLayout=vk::ImageLayout::eTransferDstOptimal;
                barrier.dstAccessMask=vk::AccessFlagBits::eTransferWrite;
                command.pipelineBarrier(
                    vk::PipelineStageFlagBits::eTopOfPipe,
                    vk::PipelineStageFlagBits::eTransfer,
                    {},{},{},{barrier});

                vk::BufferImageCopy copy{};
                copy.bufferOffset=source.offset;
                copy.imageSubresource=vk::ImageSubresourceLayers{
                    vk::ImageAspectFlagBits::eColor,0,0,1};
                copy.imageExtent=vk::Extent3D{width,height,1u};
                command.copyBufferToImage(
                    source.buffer,image.handle,
                    vk::ImageLayout::eTransferDstOptimal,{copy});

                auto src_width=width,src_height=height;
                for(std::uint32_t mip=1;mip<levels;++mip) {
                    barrier.subresourceRange=vk::ImageSubresourceRange{
                        vk::ImageAspectFlagBits::eColor,mip-1,1,0,1};
                    barrier.oldLayout=vk::ImageLayout::eTransferDstOptimal;
                    barrier.newLayout=vk::ImageLayout::eTransferSrcOptimal;
                    barrier.srcAccessMask=vk::AccessFlagBits::eTransferWrite;
                    barrier.dstAccessMask=vk::AccessFlagBits::eTransferRead;
                    command.pipelineBarrier(
                        vk::PipelineStageFlagBits::eTransfer,
                        vk::PipelineStageFlagBits::eTransfer,
                        {},{},{},{barrier});
                    const auto dst_width=std::max(1u,src_width/2u);
                    const auto dst_height=std::max(1u,src_height/2u);
                    vk::ImageBlit blit{};
                    blit.srcSubresource=vk::ImageSubresourceLayers{
                        vk::ImageAspectFlagBits::eColor,mip-1,0,1};
                    blit.srcOffsets[0]=vk::Offset3D{0,0,0};
                    blit.srcOffsets[1]=vk::Offset3D{
                        static_cast<std::int32_t>(src_width),
                        static_cast<std::int32_t>(src_height),1};
                    blit.dstSubresource=vk::ImageSubresourceLayers{
                        vk::ImageAspectFlagBits::eColor,mip,0,1};
                    blit.dstOffsets[0]=vk::Offset3D{0,0,0};
                    blit.dstOffsets[1]=vk::Offset3D{
                        static_cast<std::int32_t>(dst_width),
                        static_cast<std::int32_t>(dst_height),1};
                    command.blitImage(image.handle,
                        vk::ImageLayout::eTransferSrcOptimal,image.handle,
                        vk::ImageLayout::eTransferDstOptimal,
                        {blit},vk::Filter::eLinear);
                    src_width=dst_width;
                    src_height=dst_height;
                }
                for(std::uint32_t mip=0;mip<levels;++mip) {
                    const bool was_source=mip+1<levels;
                    barrier.subresourceRange=vk::ImageSubresourceRange{
                        vk::ImageAspectFlagBits::eColor,mip,1,0,1};
                    barrier.oldLayout=was_source?
                        vk::ImageLayout::eTransferSrcOptimal:
                        vk::ImageLayout::eTransferDstOptimal;
                    barrier.newLayout=vk::ImageLayout::eShaderReadOnlyOptimal;
                    barrier.srcAccessMask=was_source?
                        vk::AccessFlagBits::eTransferRead:
                        vk::AccessFlagBits::eTransferWrite;
                    barrier.dstAccessMask=vk::AccessFlagBits::eShaderRead;
                    command.pipelineBarrier(
                        vk::PipelineStageFlagBits::eTransfer,
                        vk::PipelineStageFlagBits::eFragmentShader,
                        {},{},{},{barrier});
                }
                if(uploaded_levels)*uploaded_levels=levels;
                return image;
            }
            catch (...)
            {
                if(image.handle)vk::image::destroy(image);
                throw;
            }
        }

        void write(vk::buffer::resource& target, const void* data,
                   std::size_t bytes, std::size_t offset = 0)
        {
            if (!data || !bytes || offset > target.size || bytes > target.size - offset)
                throw std::invalid_argument("Invalid staged buffer update");

            const auto source = stage(data,bytes);
            if (!command) command = vk::command::begin_one_time();
            ::vs::gpu::transfer::copy(command, source.buffer,
                                     target.handle, bytes,
                                     source.offset, offset);
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
