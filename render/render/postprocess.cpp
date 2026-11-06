#include "postprocess.hpp"
#include <stdexcept>
#include <utility>

namespace render {
namespace {
std::unique_ptr<gpu::image> make_image(vk::Extent2D extent,vk::Format format) {
    auto result=std::make_unique<gpu::image>();
    vk::ImageCreateInfo info{};
    info.imageType=vk::ImageType::e2D;
    info.format=format;
    info.extent=vk::Extent3D{extent.width,extent.height,1};
    info.mipLevels=1;
    info.arrayLayers=1;
    info.samples=vk::SampleCountFlagBits::e1;
    info.tiling=vk::ImageTiling::eOptimal;
    info.usage=vk::ImageUsageFlagBits::eStorage|vk::ImageUsageFlagBits::eSampled;
    result->create(info,vk::MemoryPropertyFlagBits::eDeviceLocal);
    vk::ImageViewCreateInfo view{};
    view.viewType=vk::ImageViewType::e2D;
    view.format=format;
    view.subresourceRange={vk::ImageAspectFlagBits::eColor,0,1,0,1};
    result->make_view(view);
    return result;
}
}
void postprocess_targets::create(std::size_t count,vk::Extent2D extent,
                                 vk::Format hdr_format,vk::Format ldr_format) {
    if(!slots_.empty())throw std::logic_error("postprocess targets already created");
    if(!count||!extent.width||!extent.height)
        throw std::invalid_argument("invalid postprocess extent or image count");
    auto physical=vk::physical_device::get()->handle;
    const auto required=vk::FormatFeatureFlagBits::eStorageImage |
                        vk::FormatFeatureFlagBits::eSampledImage;
    for(auto format:{hdr_format,ldr_format}) {
        const auto features=physical.getFormatProperties(format).optimalTilingFeatures;
        if((features & required)!=required)
            throw std::runtime_error("offscreen storage image format unsupported");
    }
    try {
        slots_.reserve(count);
        for(std::size_t i=0;i<count;++i) {
            slot current;
            current.hdr=make_image(extent,hdr_format);
            current.ldr=make_image(extent,ldr_format);
            slots_.push_back(std::move(current));
        }
    }catch(...) {reset();throw;}
}
postprocess_targets::attachments postprocess_targets::at(std::size_t index) const {
    const auto& s=slots_.at(index);
    return {s.hdr->handle(),s.hdr->view(),s.ldr->handle(),s.ldr->view()};
}
void postprocess_targets::reset() noexcept {slots_.clear();}
}

namespace render {
void presentation_targets::create(const std::vector<vk::Image>& images,
                                  vk::Format format,vk::Extent2D extent) {
    if(pass_ || !views_.empty())throw std::logic_error("presentation already created");
    if(images.empty()||!extent.width||!extent.height)
        throw std::invalid_argument("invalid swapchain presentation target");
    vk::AttachmentDescription attachment{};
    attachment.format=format;
    attachment.samples=vk::SampleCountFlagBits::e1;
    attachment.loadOp=vk::AttachmentLoadOp::eClear;
    attachment.storeOp=vk::AttachmentStoreOp::eStore;
    attachment.initialLayout=vk::ImageLayout::eUndefined;
    attachment.finalLayout=vk::ImageLayout::ePresentSrcKHR;
    vk::AttachmentReference color{0,vk::ImageLayout::eColorAttachmentOptimal};
    vk::SubpassDescription subpass{};
    subpass.pipelineBindPoint=vk::PipelineBindPoint::eGraphics;
    subpass.colorAttachmentCount=1;
    subpass.pColorAttachments=&color;
    vk::RenderPassCreateInfo pass_info{};
    pass_info.attachmentCount=1;
    pass_info.pAttachments=&attachment;
    pass_info.subpassCount=1;
    pass_info.pSubpasses=&subpass;
    try {
        pass_=vk::render_pass::create(pass_info);
        for(auto image:images) {
            vk::ImageViewCreateInfo view{};
            view.image=image;
            view.viewType=vk::ImageViewType::e2D;
            view.format=format;
            view.subresourceRange={vk::ImageAspectFlagBits::eColor,0,1,0,1};
            auto handle=vk::image_view::create(view);
            views_.push_back(handle);
            vk::FramebufferCreateInfo info{};
            info.renderPass=pass_;
            info.attachmentCount=1;
            info.pAttachments=&views_.back();
            info.width=extent.width;
            info.height=extent.height;
            info.layers=1;
            buffers_.push_back(vk::framebuffer::create(info));
        }
    }catch(...){reset();throw;}
}
void presentation_targets::reset() noexcept {
    for(auto framebuffer:buffers_)if(framebuffer)vk::framebuffer::destroy(framebuffer);
    buffers_.clear();
    for(auto view:views_)if(view)vk::image_view::destroy(view);
    views_.clear();
    if(pass_)vk::render_pass::destroy(pass_);
    pass_=vk::RenderPass{};
}
}
