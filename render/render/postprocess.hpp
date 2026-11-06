#pragma once
#include <gpu/image.hpp>
#include <vk/vk.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace render {
// Owns per-swapchain-image HDR and LDR compute targets. The scene-specific
// lighting and tone shaders remain in the consumer. Resize only at GPU idle.
class postprocess_targets {
public:
    struct attachments {
        vk::Image hdr{};
        vk::ImageView hdr_view{};
        vk::Image ldr{};
        vk::ImageView ldr_view{};
    };
    postprocess_targets()=default;
    ~postprocess_targets(){reset();}
    postprocess_targets(const postprocess_targets&)=delete;
    postprocess_targets& operator=(const postprocess_targets&)=delete;
    void create(std::size_t images,vk::Extent2D extent,
                vk::Format hdr_format=vk::Format::eR16G16B16A16Sfloat,
                vk::Format ldr_format=vk::Format::eR8G8B8A8Unorm);
    attachments at(std::size_t index) const;
    std::size_t size() const noexcept {return slots_.size();}
    void reset() noexcept;
private:
    struct slot {std::unique_ptr<gpu::image> hdr,ldr;};
    std::vector<slot> slots_;
};
}

namespace render {
// Presentation pass and framebuffers for externally owned swapchain images.
class presentation_targets {
public:
    presentation_targets()=default;
    ~presentation_targets(){reset();}
    presentation_targets(const presentation_targets&)=delete;
    presentation_targets& operator=(const presentation_targets&)=delete;
    void create(const std::vector<vk::Image>& swapchain_images,
                vk::Format format,vk::Extent2D extent);
    vk::RenderPass pass() const noexcept {return pass_;}
    vk::Framebuffer framebuffer(std::size_t index) const {return buffers_.at(index);}
    void reset() noexcept;
private:
    vk::RenderPass pass_{};
    std::vector<vk::ImageView> views_;
    std::vector<vk::Framebuffer> buffers_;
};
}
