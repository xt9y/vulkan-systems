#pragma once
#include <gpu/resource.hpp>
#include <vk/vk.hpp>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace render
{
    // Generic device-local, sampled 2D framebuffer attachments.
    // This owns images and views; render-pass/framebuffer lifetime stays separate.
    class attachments
    {
        struct image_view
        {
            gpu::owned_image image;
            vk::ImageView view{};
            image_view()=default;
            image_view(const image_view&)=delete;
            image_view& operator=(const image_view&)=delete;
            ~image_view() noexcept
            {
                if(view)
                {
                    try{vk::image_view::destroy(view);}catch(...){}
                }
            }
        };
        std::vector<std::unique_ptr<image_view>> colors_;
        std::unique_ptr<image_view> depth_;

        static std::unique_ptr<image_view> make(vk::Extent2D extent,
                                                vk::Format format,bool depth)
        {
            if(!extent.width || !extent.height || format==vk::Format::eUndefined)
                throw std::invalid_argument("Invalid framebuffer attachment");
            const auto required=depth?
                vk::FormatFeatureFlags(vk::FormatFeatureFlagBits::eDepthStencilAttachment |
                                       vk::FormatFeatureFlagBits::eSampledImage):
                vk::FormatFeatureFlags(vk::FormatFeatureFlagBits::eColorAttachment |
                                       vk::FormatFeatureFlagBits::eSampledImage);
            const auto features=vk::physical_device::get()->handle
                .getFormatProperties(format).optimalTilingFeatures;
            if((features&required)!=required)
                throw std::runtime_error("Unsupported sampled framebuffer format");

            auto result=std::make_unique<image_view>();
            vk::ImageCreateInfo info{};
            info.imageType=vk::ImageType::e2D;
            info.format=format;
            info.extent=vk::Extent3D{extent.width,extent.height,1};
            info.mipLevels=1;
            info.arrayLayers=1;
            info.samples=vk::SampleCountFlagBits::e1;
            info.tiling=vk::ImageTiling::eOptimal;
            info.usage=depth?
                vk::ImageUsageFlags(vk::ImageUsageFlagBits::eDepthStencilAttachment |
                                    vk::ImageUsageFlagBits::eSampled):
                vk::ImageUsageFlags(vk::ImageUsageFlagBits::eColorAttachment |
                                    vk::ImageUsageFlagBits::eSampled);
            result->image=gpu::create_image(info,vk::MemoryPropertyFlagBits::eDeviceLocal);
            vk::ImageViewCreateInfo view{};
            view.image=result->image.handle;
            view.viewType=vk::ImageViewType::e2D;
            view.format=format;
            view.subresourceRange={
                depth?vk::ImageAspectFlagBits::eDepth:vk::ImageAspectFlagBits::eColor,
                0,1,0,1
            };
            result->view=vk::image_view::create(view);
            return result;
        }
    public:
        attachments()=default;
        attachments(const attachments&)=delete;
        attachments& operator=(const attachments&)=delete;
        attachments(attachments&&)=default;
        attachments& operator=(attachments&&)=default;

        void create(vk::Extent2D extent,const std::vector<vk::Format>& color_formats,
                    vk::Format depth_format=vk::Format::eUndefined)
        {
            attachments result;
            result.colors_.reserve(color_formats.size());
            for(auto format:color_formats)
                result.colors_.push_back(make(extent,format,false));
            if(depth_format!=vk::Format::eUndefined)
                result.depth_=make(extent,depth_format,true);
            *this=std::move(result);
        }
        void reset() noexcept
        {
            depth_.reset();
            colors_.clear();
        }
        std::size_t color_count() const noexcept {return colors_.size();}
        vk::ImageView color(std::size_t index) const {return colors_.at(index)->view;}
        vk::ImageView depth() const noexcept {return depth_?depth_->view:vk::ImageView{};}
    };
}
