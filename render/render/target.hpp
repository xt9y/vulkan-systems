#pragma once
#include <vk/vk.hpp>
#include <optional>
#include <vector>

namespace render
{
    // One generic graphics subpass with arbitrary color attachments and
    // optional depth. Attachments/images are owned by the caller.
    struct target_description
    {
        std::vector<vk::AttachmentDescription> attachments;
        std::vector<vk::AttachmentReference> colors;
        std::optional<vk::AttachmentReference> depth;
        std::vector<vk::ImageView> views;
        vk::Extent2D extent{};
        std::uint32_t layers=1;
    };

    class target
    {
    public:
        target()=default;
        ~target(){reset();}
        target(const target&)=delete;
        target& operator=(const target&)=delete;
        void create(const target_description& description);
        // Build a framebuffer for a render pass owned elsewhere. Caller must
        // keep the render pass and attachment views alive through reset().
        void create_framebuffer(vk::RenderPass pass,
                                const std::vector<vk::ImageView>& views,
                                vk::Extent2D extent,std::uint32_t layers=1);
        void reset();
        vk::RenderPass pass() const{return pass_;}
        vk::Framebuffer framebuffer() const{return framebuffer_;}
        vk::Extent2D extent() const{return extent_;}
    private:
        vk::RenderPass pass_{};
        vk::Framebuffer framebuffer_{};
        vk::Extent2D extent_{};
        bool owns_pass_=false;
    };
}
