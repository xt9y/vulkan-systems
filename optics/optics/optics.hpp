#pragma once
#include <vk/vk.hpp>
#include <gpu/descriptors.hpp>
#include <gpu/buffer.hpp>
#include <gpu/resource.hpp>
#include <render/pipeline.hpp>
#include <array>
#include <cstdint>
#include <memory>
#include <vector>

namespace optics {
// Source images are owned by the host. Recreate when its swapchain or
// material targets change; recording requires an active compatible render pass.
struct input {
    vk::ImageView lit{}, opaque_depth{};
    vk::ImageView normal{}, tint{}, material{}, material_depth{};
    vk::ImageView probes{},hdr{},reflection{};
};
// Triangle inputs and animation updates; BVH storage belongs to VulkanOptics.
struct surface {
    using vec3=vk::util::math::vec3;
    vec3 origin{},u{},v{},normal{};
    vec3 albedo{1.f,1.f,1.f},emission{};
    std::uint32_t pattern=0,optical_type=0;
    float optical_strength=0.f,roughness=0.f,ior=1.52f;
};
struct geometry_update {std::uint32_t face=0;surface value{};};
// Per-face UVs remain aligned with the optics-owned BVH.
// The texture atlas is built and owned by VulkanOptics, not the game.
struct face_uv {
    std::array<float,2> a{},b{},c{};
    std::uint32_t texture_id=0;
    std::uint32_t metallic_roughness_texture=0,normal_texture=0;
    std::uint32_t occlusion_texture=0,emissive_texture=0;
    float alpha_cutoff=0.f,base_alpha=1.f;
};
// Each entry corresponds to one BVH triangle face.
// Smooth normals are in world space and share the same vertex order as UVs.
struct face_normals {
    std::array<float,4> a{},b{},c{};
};
struct face_normal_update {
    std::uint32_t face=0;
    face_normals value{};
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
    compositor();
    ~compositor();
    compositor(const compositor&)=delete;
    compositor& operator=(const compositor&)=delete;
    void create(vk::RenderPass pass, const std::vector<input>& frames,
                const char* vertex="build/shaders/pcloud_fullscreen.vert.spv",
                const char* fragment="build/shaders/optics_compose.frag.spv",
                const char* mirror_fragment="build/shaders/optics_compose_mirror.frag.spv",
                const char* planar_fragment="build/shaders/optics_compose_planar.frag.spv",
                const char* glass_fragment="build/shaders/optics_compose_glass.frag.spv");
    // Build and own the acceleration structure; submissions must be idle
    // while replacing scene geometry or its material buffers.
    void set_scene(const std::vector<surface>&,const std::vector<light>&,
                   bool contains_glass=true);
    void update_geometry_frame(std::uint32_t,
                               const std::vector<geometry_update>&);
    // Lighting changes need no new BVH or surface atlas.
    void set_lights(const std::vector<light>&);
    void set_textures(const std::vector<face_uv>&,
                      const std::vector<texture_rgba>&);
    void set_normals(const std::vector<face_normals>&);
    void update_normals(std::uint32_t,
                        const std::vector<face_normal_update>&);
    void set_probe_space(const std::array<float,4>&);
    void set_probe_views(const std::vector<vk::ImageView>&);
    // Only emit fragments inside rasterized material bounds.
    void record(vk::CommandBuffer command,std::uint32_t frame,
                const camera& view,
                const std::vector<vk::Rect2D>& glass_regions,
                const std::vector<vk::Rect2D>& mirror_regions,
                vk::Extent2D extent) const;
    void reset() noexcept;
private:
    struct acceleration;
    std::unique_ptr<acceleration> acceleration_;
    gpu::descriptors descriptors_;
    render::pipeline pipeline_,mirror_pipeline_,planar_pipeline_,glass_pipeline_;
    gpu::buffer empty_,lights_,face_uv_,face_normals_,probe_space_;
    struct animated_normals {
        gpu::buffer buffer;
        std::vector<face_normals> faces;
    };
    std::vector<face_normals> original_normals_;
    std::vector<std::unique_ptr<animated_normals>> frame_normals_;
    std::uint32_t face_count_=0;
    bool contains_glass_=true,has_mirror_=false;
    gpu::owned_image atlas_;
    vk::ImageView atlas_view_{};
    vk::Sampler sampler_{},hdr_sampler_{},atlas_sampler_{};
    vk::PipelineLayout layout_{};
    std::uint32_t frame_count_=0;
};
} // namespace optics
