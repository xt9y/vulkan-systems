#pragma once
#include <vk/vk.hpp>
#include <cstdint>
#include <memory>

namespace lumen {

// Lumen-style rendering API: Cornell demo with material cards, BVH direct
// lighting and budgeted world-space diffuse radiance probes. Reflections pending.
enum class demo_scene : std::uint8_t {
    none,
    cornell_box
};

struct settings {
    demo_scene demo=demo_scene::cornell_box;
    std::uint32_t probe_updates_per_frame=64;
    std::uint32_t rays_per_probe=4;
    // Internal raster resolution scale, capped at 1280 pixels wide by Lumen.
    float tracing_scale=1.0f;
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

// Indirect/reflection outputs stay null until the corresponding stages exist.
struct outputs {
    vk::Image diffuse{};
    vk::Image reflections{};
};

// Measured previous-frame GPU time; unavailable on queues without timestamps.
// Excludes swapchain present and CPU submission/wait time.
struct performance {
    float material_cache_ms=0.f;
    float radiance_cache_ms=0.f;
    float raster_ms=0.f;
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

    // Call after VulkanEngine device/swapchain setup. Compatible with host pass.
    void create(vk::RenderPass presentation_pass, vk::Format presentation_format,
                vk::Extent2D extent, const settings& options={});
    void resize(vk::RenderPass presentation_pass, vk::Format presentation_format,
                vk::Extent2D extent);
    void set_scene(const scene& geometry);
    void set_lighting(const lighting& lights);
    void set_gbuffer(const gbuffer& surfaces);
    void invalidate(); // Future cache invalidation on scene/light changes.

    // record() runs before the host render pass; compose() runs inside it.
    // Records the Cornell direct-light pass and its presentation composite.
    void record(vk::CommandBuffer command, const frame& current_frame);
    void compose(vk::CommandBuffer command);
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
