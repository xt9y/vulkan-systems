#include "target.hpp"
#include <stdexcept>

namespace render
{
    void target::create(const target_description& description)
    {
        if(pass_ || framebuffer_)throw std::logic_error("target already created");
        if(description.attachments.empty() ||
           description.attachments.size()!=description.views.size() ||
           !description.extent.width || !description.extent.height ||
           !description.layers)
            throw std::invalid_argument("invalid render target attachments");
        for(const auto& color:description.colors)
            if(color.attachment>=description.attachments.size())
                throw std::invalid_argument("invalid color attachment index");
        if(description.depth &&
           description.depth->attachment>=description.attachments.size())
            throw std::invalid_argument("invalid depth attachment index");

        vk::SubpassDescription subpass{};
        subpass.pipelineBindPoint=vk::PipelineBindPoint::eGraphics;
        subpass.colorAttachmentCount=std::uint32_t(description.colors.size());
        subpass.pColorAttachments=description.colors.empty()?
            nullptr:description.colors.data();
        subpass.pDepthStencilAttachment=description.depth?&*description.depth:nullptr;

        vk::RenderPassCreateInfo pass_info{};
        pass_info.attachmentCount=std::uint32_t(description.attachments.size());
        pass_info.pAttachments=description.attachments.data();
        pass_info.subpassCount=1;
        pass_info.pSubpasses=&subpass;
        try
        {
            pass_=vk::render_pass::create(pass_info);
            owns_pass_=true;
            vk::FramebufferCreateInfo frame_info{};
            frame_info.renderPass=pass_;
            frame_info.attachmentCount=std::uint32_t(description.views.size());
            frame_info.pAttachments=description.views.data();
            frame_info.width=description.extent.width;
            frame_info.height=description.extent.height;
            frame_info.layers=description.layers;
            framebuffer_=vk::framebuffer::create(frame_info);
            extent_=description.extent;
        }
        catch(...)
        {
            reset();
            throw;
        }
    }

    void target::create_framebuffer(vk::RenderPass pass,
                                    const std::vector<vk::ImageView>& views,
                                    vk::Extent2D extent,std::uint32_t layers)
    {
        if(pass_||framebuffer_)throw std::logic_error("target already created");
        if(!pass||views.empty()||!extent.width||!extent.height||!layers)
            throw std::invalid_argument("invalid external render pass/framebuffer");
        vk::FramebufferCreateInfo info{};
        info.renderPass=pass;
        info.attachmentCount=std::uint32_t(views.size());
        info.pAttachments=views.data();
        info.width=extent.width;
        info.height=extent.height;
        info.layers=layers;
        // Do not take ownership of the caller's pass even on failure.
        framebuffer_=vk::framebuffer::create(info);
        pass_=pass;
        extent_=extent;
        owns_pass_=false;
    }

    void target::reset()
    {
        if(framebuffer_)vk::framebuffer::destroy(framebuffer_);
        framebuffer_=vk::Framebuffer{};
        if(pass_&&owns_pass_)vk::render_pass::destroy(pass_);
        owns_pass_=false;
        pass_=vk::RenderPass{};
        extent_=vk::Extent2D{};
    }
}
