#pragma once
#include <vk/vk.hpp>
#include <cstdint>
#include <memory>

namespace lumen {

// Cornell Lumen prototype: separately lit surface cards, world radiance
// probes and directional screen-radiance final gather with scene fallback.
enum class demo_scene : std::uint8_t {
    none,
    cornell_box
};

struct settings {
    demo_scene demo=demo_scene::cornell_box;
    std::uint32_t probe_updates_per_frame=64;
    std::uint32_t rays_per_probe=8;
    // Internal raster resolution scale, capped at 1280 pixels wide by Lumen.
    float tracing_scale=1.0f;
    // Depth-screen ray tracing, with a geometry-BVH fallback on misses.
    float screen_trace_distance=12.0f;
    std::uint32_t screen_rays_per_probe=4; // 1..8 directional rays/frame
    bool full_resolution_outputs=false; // optional compute reconstruction
};

struct camera {
    vk::util::math::vec3 position{};
    float yaw=0.f;
    float pitch=0.f;
    float vertical_fov=39.f;
};

struct frame {
    camera view{};
    std::uint64_t index=0;
    float delta_seconds=0.f;
};

// Externally owned scene data. VulkanLumen never takes ownership of these handles.
// A self-contained demo scene can instead be selected through settings::demo.
struct scene {
    vk::Buffer vertices{};
    vk::Buffer indices{};
    vk::Buffer instances{};
    vk::Buffer materials{};
    std::uint32_t vertex_count=0;
    std::uint32_t index_count=0;
    std::uint32_t instance_count=0;
};

struct lighting {
    vk::Buffer lights{};
    std::uint32_t count=0;
};

// Optional primary rasterization guides, owned by the calling renderer.
struct gbuffer {
    vk::Image depth{};
    vk::Image normal{};
    vk::Image albedo{};
    vk::Image roughness{};
    vk::Image motion{};
    vk::Extent2D extent{};
};

// Borrowed, screen-probe-resolution outputs for host-renderer integration.
// The RGB channels of diffuse and reflections hold their independent
// lighting contributions; alpha stores the Cornell face ID (zero = invalid).
// metadata.xyz is the surface world-space history anchor and metadata.w is
// camera-space depth. scene_color is direct+emissive HDR and scene_depth is
// the internally rasterized D32 surface depth.
//
// All image/image-view handles remain owned by VulkanLumen. They are valid
// only while the system is alive and until resize(), reset() or create-time
// replacement. Call result() after record(); the caller must synchronize the
// recorded GPU work before sampling the images on another queue.
// Lighting images are in eShaderReadOnlyOptimal after record(); scene_depth
// is in eDepthStencilReadOnlyOptimal. No full-resolution lighting output is
// implied by this API unless full_resolution_outputs is enabled. In that
// case diffuse/reflection are at scene_extent; metadata remains at
// probe_extent. Use lighting_extent to distinguish the returned size.
struct outputs {
    vk::Image diffuse{},reflections{},metadata{},scene_color{},scene_depth{};
    vk::ImageView diffuse_view{},reflections_view{},metadata_view{};
    vk::ImageView scene_color_view{},scene_depth_view{};
    vk::Extent2D probe_extent{},scene_extent{},lighting_extent{};
    bool full_resolution=false;
    vk::Format radiance_format=vk::Format::eUndefined;
    vk::Format depth_format=vk::Format::eUndefined;
    vk::ImageLayout color_layout=vk::ImageLayout::eUndefined;
    vk::ImageLayout depth_layout=vk::ImageLayout::eUndefined;
    bool available=false;
};

// Measured previous-frame GPU time; unavailable on queues without timestamps.
// Excludes swapchain present and CPU submission/wait time.
struct performance {
    float material_cache_ms=0.f;
    float radiance_cache_ms=0.f;
    float raster_ms=0.f;
    float screen_probes_ms=0.f;
    float composite_ms=0.f;
    float total_ms=0.f;
    bool supported=false;
    bool valid=false;
};

class system {
public:
    system();
    ~system();
    system(const system&)=delete;
    system& operator=(const system&)=delete;
    system(system&&) noexcept;
    system& operator=(system&&) noexcept;

    // Call after VulkanWrapper device/swapchain setup. Compatible with host pass.
    void create(vk::RenderPass presentation_pass, vk::Format presentation_format,
                vk::Extent2D extent, const settings& options={});
    void resize(vk::RenderPass presentation_pass, vk::Format presentation_format,
                vk::Extent2D extent);
    void set_scene(const scene& geometry);
    void set_lighting(const lighting& lights);
    void set_gbuffer(const gbuffer& surfaces);
    void invalidate(); // Rebuild Cornell material, direct/indirect and probe caches.

    // record() runs before the host render pass; compose() runs inside it.
    // Records the internal demo, all caches and screen radiance integration.
    void record(vk::CommandBuffer command, const frame& current_frame);
    void compose(vk::CommandBuffer command);
    // Result handles become available after record() and are invalidated
    // by resize()/reset(). The host does not own the image or image views.
    outputs result() const noexcept;
    performance gpu_performance() const noexcept;
    bool implemented() const noexcept; // True once the Cornell demo can render.
    const char* status() const noexcept; // Rendering milestone for terminal logs.
    void reset() noexcept;

private:
    struct implementation;
    std::unique_ptr<implementation> impl_;
};

} // namespace lumen
