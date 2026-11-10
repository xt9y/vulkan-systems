#pragma once
#include <vk/vk.hpp>
#include <gpu/descriptors.hpp>
#include <gpu/buffer.hpp>
#include <gpu/resource.hpp>
#include <render/pipeline.hpp>
#include <array>
#include <cstdint>
#include <vector>

namespace optics {
// Source images are owned by the host. Recreate when its swapchain or
// material targets change; recording requires an active compatible render pass.
struct input {
    vk::ImageView lit{}, opaque_depth{}, hdr{};
    vk::ImageView normal{}, tint{}, material{}, material_depth{};
};
struct geometry {
    vk::Buffer faces{},nodes{},triangles{},scene_info{};
    vk::Buffer photons{},offsets{};
    std::uint32_t face_count=0;
};
// Per-BVH-face UV triangles stay index-aligned with VulkanPMap surfaces.
// The texture atlas is built and owned by VulkanOptics, not the game.
struct face_uv {
    std::array<float,2> a{},b{},c{};
    std::uint32_t texture_id=0;
    float alpha_cutoff=0.f;
};
// Each entry corresponds to one BVH face, including transport emitters.
// Smooth normals are in world space and share the same vertex order as UVs.
struct face_normals {
    std::array<float,4> a{},b{},c{};
};
struct texture_rgba {
    std::uint32_t id=0,width=0,height=0;
    std::vector<std::uint8_t> pixels;
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
    void set_textures(const std::vector<face_uv>&,
                      const std::vector<texture_rgba>&);
    void set_normals(const std::vector<face_normals>&);
    // Fullscreen resolve masks all non-optical fragments before BVH tracing.
    void record(vk::CommandBuffer command,std::uint32_t frame,
                const camera& view,vk::Extent2D extent) const;
    void reset() noexcept;
private:
    gpu::descriptors descriptors_;
    render::pipeline pipeline_;
    gpu::buffer empty_,lights_,face_uv_,face_normals_;
    std::uint32_t face_count_=0;
    gpu::owned_image atlas_;
    vk::ImageView atlas_view_{};
    vk::Sampler sampler_{},atlas_sampler_{};
    vk::PipelineLayout layout_{};
    std::uint32_t frame_count_=0;
};
} // namespace optics
