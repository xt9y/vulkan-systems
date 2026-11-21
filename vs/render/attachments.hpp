#pragma once
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>
#include <vk/vk.hpp>
#include <vs/gpu/image.hpp>

namespace vs::render
{
// Same owning GPU image and view used by texture caches and render passes.
// No second framebuffer-specific resource owner or manual view cleanup.
class attachments
{
    std::vector<::vs::gpu::image> colors_;
    std::optional<::vs::gpu::image> depth_;

    static ::vs::gpu::image make(vk::Extent2D extent,
                                 vk::Format format, bool depth)
    {
        if (!extent.width || !extent.height || format == vk::Format::eUndefined)
            throw std::invalid_argument("Invalid framebuffer attachment");

        const auto required = depth ? vk::FormatFeatureFlags(vk::FormatFeatureFlagBits::eDepthStencilAttachment |
                                                             vk::FormatFeatureFlagBits::eSampledImage)
                                    : vk::FormatFeatureFlags(vk::FormatFeatureFlagBits::eColorAttachment |
                                                             vk::FormatFeatureFlagBits::eSampledImage);
        const auto features = vk::physical_device::get()->handle.getFormatProperties(format).optimalTilingFeatures;

        if ((features & required) != required)
            throw std::runtime_error("Unsupported sampled framebuffer format");
        ::vs::gpu::image result;
        vk::ImageCreateInfo info{};
        info.imageType = vk::ImageType::e2D;
        info.format = format;
        info.extent = vk::Extent3D{extent.width, extent.height, 1};
        info.mipLevels = info.arrayLayers = 1;
        info.samples = vk::SampleCountFlagBits::e1;
        info.tiling = vk::ImageTiling::eOptimal;
        info.usage = depth ? vk::ImageUsageFlags(vk::ImageUsageFlagBits::eDepthStencilAttachment |
                                                 vk::ImageUsageFlagBits::eSampled)
                           : vk::ImageUsageFlags(vk::ImageUsageFlagBits::eColorAttachment |
                                                 vk::ImageUsageFlagBits::eSampled);
        result.create(info, vk::MemoryPropertyFlagBits::eDeviceLocal);
        vk::ImageViewCreateInfo view{};
        view.viewType = vk::ImageViewType::e2D;
        view.format = format;
        view.subresourceRange = {
            depth ? vk::ImageAspectFlagBits::eDepth : vk::ImageAspectFlagBits::eColor,
            0, 1, 0, 1
        };

        result.make_view(view);

        return result;
    }

public:
    attachments() = default;
    attachments(const attachments &) = delete;
    attachments &operator=(const attachments &) = delete;
    attachments(attachments &&) = default;
    attachments &operator=(attachments &&) = default;
    void create(vk::Extent2D extent, const std::vector<vk::Format> &color_formats,
                vk::Format depth_format = vk::Format::eUndefined)
    {
        attachments next;
        next.colors_.reserve(color_formats.size());

        for (auto format : color_formats)
            next.colors_.push_back(make(extent, format, false));

        if (depth_format != vk::Format::eUndefined)
            next.depth_.emplace(make(extent, depth_format, true));
        *this = std::move(next);
    }

    void reset() noexcept
    {
        depth_.reset();
        colors_.clear();
    }

    std::size_t color_count() const noexcept { return colors_.size(); }
    vk::ImageView color(std::size_t index) const { return colors_.at(index).view(); }
    vk::ImageView depth() const noexcept { return depth_ ? depth_->view() : vk::ImageView{}; }
};
} // namespace vs::render
