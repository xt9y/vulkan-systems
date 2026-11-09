#pragma once
#include <vk/vk.hpp>
#include <gpu/descriptors.hpp>
#include <gpu/buffer.hpp>
#include <render/pipeline.hpp>
#include <array>
#include <cstdint>
#include <vector>

namespace optics {
// Source images are owned by the host. Recreate when its swapchain or
// material targets change; recording requires an active compatible render pass.
struct input {
    vk::ImageView lit{}, opaque_depth{};
    vk::ImageView normal{}, tint{}, material{}, material_depth{};
};
struct geometry {
    vk::Buffer faces{},nodes{},triangles{},scene_info{};
    vk::Buffer photons{},offsets{};
};
struct light {
    vk::util::math::vec3 position{};
    float radius=1.f;
    vk::util::math::vec3 color{1.f,1.f,1.f};
    float strength=1.f;
};
struct camera {
    vk::util::math::mat4 view_projection;
    vk::util::math::vec4 eye, right, up, forward;
};
static_assert(sizeof(camera)==128);

class compositor {
public:
    compositor()=default;
    ~compositor(){reset();}
    compositor(const compositor&)=delete;
    compositor& operator=(const compositor&)=delete;
    void create(vk::RenderPass pass, const std::vector<input>& frames,
                const char* vertex="build/shaders/pcloud_fullscreen.vert.spv",
                const char* fragment="build/shaders/optics_compose.frag.spv");
    // Share the photon mapper's static GPU BVH; the mapper must outlive
    // submitted rendering work. Empty geometry disables off-screen rays.
    void set_scene(geometry,const std::vector<light>&);
    void record(vk::CommandBuffer command,std::uint32_t frame,
                const camera& view) const;
    void reset() noexcept;
private:
    gpu::descriptors descriptors_;
    render::pipeline pipeline_;
    gpu::buffer empty_,lights_;
    vk::Sampler sampler_{};
    vk::PipelineLayout layout_{};
    std::uint32_t frame_count_=0;
};
} // namespace optics
