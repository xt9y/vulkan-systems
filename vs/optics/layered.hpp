#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include <vk/vk.hpp>
#include <vs/gpu/buffer.hpp>
#include <vs/gpu/descriptors.hpp>
#include <vs/gpu/resource.hpp>
#include <vs/render/pipeline.hpp>
#include <vs/scene/scene.hpp>

namespace vs::optics
{
struct face_uv
{
    std::array<float, 2> a{}, b{}, c{};
    std::uint32_t texture_id = 0;
    std::uint32_t metallic_roughness_texture = 0, normal_texture = 0;
    std::uint32_t occlusion_texture = 0, emissive_texture = 0;

    float alpha_cutoff = 0.f, base_alpha = 1.f;
};

struct face_normals
{
    std::array<float, 4> a{}, b{}, c{};
};

struct face_normal_update
{
    std::uint32_t face = 0;
    face_normals value{};
};

struct camera
{
    vk::util::math::mat4 view_projection;
    vk::util::math::vec4 eye, right, up, forward;
};

static_assert(sizeof(camera) == 128);
} // namespace vs::optics

namespace vs::pcloud
{
struct shadow_geometry;
}

namespace vs::optics::layered
{
// Source images are owned by the host. Recreate when its swapchain or
// material targets change; recording requires an active compatible render pass.
struct input
{
    vk::ImageView lit{}, opaque_depth{};
    vk::ImageView normal{}, tint{}, material{}, material_depth{};
    vk::ImageView probes{}, hdr{}, reflection{};
    // A second optical layer in the primary view (depth peeled), plus
    // glass visible to the reflected camera, all owned by the host.
    vk::ImageView peel_normal{}, peel_tint{}, peel_material{}, peel_depth{};
    vk::ImageView reflected_normal{}, reflected_tint{}, reflected_material{};
    vk::ImageView reflected_depth{}, reflected_opaque_depth{};
};
// Triangle inputs and animation updates; BVH storage belongs to VulkanOptics.
// Reuse canonical VS geometry, materials, camera and texture records.
using surface = ::vs::scene::surface;

struct geometry_update
{
    std::uint32_t face = 0;
    surface value{};
};

using face_uv = ::vs::optics::face_uv;
using face_normals = ::vs::optics::face_normals;
using face_normal_update = ::vs::optics::face_normal_update;
using texture_rgba = ::vs::scene::texture;
using light = ::vs::scene::point_light;
using camera = ::vs::optics::camera;

class compositor
{
public:
    compositor();
    ~compositor();
    compositor(const compositor &) = delete;
    compositor &operator=(const compositor &) = delete;
    void create(vk::RenderPass pass, const std::vector<input> &frames,
                const char *vertex = "build/shaders/pcloud_fullscreen.vert.spv",
                const char *fragment = "build/shaders/optics_legacy_compose.frag.spv",
                const char *mirror_fragment = "build/shaders/optics_legacy_compose_mirror.frag.spv",
                const char *planar_fragment = "build/shaders/optics_legacy_compose_planar.frag.spv",
                const char *glass_fragment = "build/shaders/optics_legacy_compose_glass.frag.spv");
    // Build and own the acceleration structure; submissions must be idle
    // while replacing scene geometry or its material buffers.
    void set_scene(const std::vector<surface> &, const std::vector<light> &,
                   bool contains_glass = true);
    void update_geometry_frame(std::uint32_t,
                               const std::vector<geometry_update> &);
    // Lighting changes need no new BVH or surface atlas.
    void set_lights(const std::vector<light> &);
    // Borrows the exact opaque per-light shadow BVHs owned by PCloud.
    // Rebind after PCloud scene synchronization (GPU submissions idle).
    void set_opaque_shadows(const ::vs::pcloud::shadow_geometry &);
    // Stage replacement texture data while the active atlas remains usable.
    // Source texture storage must stay alive until the final row is queued.
    void begin_textures(const std::vector<face_uv> &,
                        const std::vector<texture_rgba> &);
    bool advance_textures(std::size_t byte_budget = 2u * 1024u * 1024u);
    void commit_textures();
    void cancel_textures() noexcept;
    void set_textures(const std::vector<face_uv> &,
                      const std::vector<texture_rgba> &);
    void set_normals(const std::vector<face_normals> &);
    void update_normals(std::uint32_t,
                        const std::vector<face_normal_update> &);
    void set_probe_space(const std::array<float, 4> &);
    void set_probe_views(const std::vector<vk::ImageView> &);
    // Only emit fragments inside rasterized material bounds.
    void record(vk::CommandBuffer command, std::uint32_t frame,
                const camera &view,
                const std::vector<vk::Rect2D> &glass_regions,
                const std::vector<vk::Rect2D> &mirror_regions,
                vk::Extent2D extent) const;
    // Compatibility with projects that still supply a shared region set.
    void record(vk::CommandBuffer command, std::uint32_t frame,
                const camera &view, const std::vector<vk::Rect2D> &regions,
                vk::Extent2D extent) const;
    void reset() noexcept;

private:
    struct acceleration;
    struct texture_upload;
    std::unique_ptr<acceleration> acceleration_;
    std::unique_ptr<texture_upload> texture_upload_;
    gpu::descriptors descriptors_;
    render::pipeline pipeline_, mirror_pipeline_, planar_pipeline_, glass_pipeline_;
    gpu::buffer empty_, lights_, face_uv_, face_normals_, probe_space_;
    struct animated_normals
    {
        gpu::buffer buffer;
        std::vector<face_normals> faces;
    };

    std::vector<face_normals> original_normals_;
    std::vector<std::unique_ptr<animated_normals>> frame_normals_;
    std::uint32_t face_count_ = 0;

    bool contains_glass_ = true, has_mirror_ = false;

    gpu::owned_image atlas_;
    vk::ImageView atlas_view_{};
    vk::Sampler sampler_{}, hdr_sampler_{}, atlas_sampler_{};
    vk::PipelineLayout layout_{};
    std::uint32_t frame_count_ = 0;
};
} // namespace vs::optics::layered
