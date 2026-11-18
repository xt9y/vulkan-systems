#pragma once
#include <vk/vk.hpp>
#include <vs/scene/scene.hpp>
#include <vs/spatial/world.hpp>
#include <span>
#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>
namespace vs::pcloud
{
    using vec3=::vs::scene::vec3;
    using light=::vs::scene::point_light;
    using collision_box=::vs::scene::collision_box;
    using triangle=::vs::scene::triangle;
    using surface=::vs::scene::surface;
    // The same conservative light-range test as the BVH builder. Hosts
    // may use it while flattening millions of meshes to avoid creating
    // irrelevant triangle inputs; pc still validates all supplied ones.
    inline bool relevant_triangle(const triangle& t,
                                  const std::vector<light>& lights)
    {
        using namespace vk::util::math;
        const auto& a=t.a;
        const auto& b=t.b;
        const auto& c=t.c;
        const vec3 lo{std::min({a.x,b.x,c.x}),
                      std::min({a.y,b.y,c.y}),
                      std::min({a.z,b.z,c.z})};
        const vec3 hi{std::max({a.x,b.x,c.x}),
                      std::max({a.y,b.y,c.y}),
                      std::max({a.z,b.z,c.z})};
        for(const auto& light:lights)
        {
            const vec3 near{
                std::clamp(light.position.x,lo.x,hi.x),
                std::clamp(light.position.y,lo.y,hi.y),
                std::clamp(light.position.z,lo.z,hi.z)
            };
            if(light.enabled && light.strength>0.f &&
               length_squared(near-light.position)<=light.radius*light.radius)
                return true;
        }
        return false;
    }

    using camera_frame=::vs::scene::camera;
    struct view_basis {
        vec3 eye{},forward{},right{},up{};
        float fov=1.0471976f;
    };
    // Shader contract: {albedo, normal, material, depth}. All views must
    // already be shader-readable when ::vs::pcloud::record is called. Packed normals
    // are N*0.5+0.5; material channels are roughness, metallic, AO.
    using gbuffer=std::array<vk::ImageView,4>;
    struct presentation_target
    {
        std::vector<vk::Image> images; // Externally owned swapchain images.
        vk::Format format=vk::Format::eUndefined;
        vk::Extent2D extent{};
    };
    struct configuration
    {
        std::string cloud_shader="build/shaders/pcloud_cloud.comp.spv";
        std::string gi_shader="build/shaders/pcloud_gi.comp.spv";
        std::string lighting_shader="build/shaders/pcloud_light.comp.spv";
        std::string tonemap_shader="build/shaders/pcloud_tone.comp.spv";
        std::string fullscreen_shader="build/shaders/pcloud_fullscreen.vert.spv";
        std::string presentation_shader="build/shaders/pcloud_present.frag.spv";

        static configuration from_directory(const std::string& dir)
        {
            const std::string base=dir.empty()?std::string("."):dir;
            configuration value{};
            value.cloud_shader=base+"/pcloud_cloud.comp.spv";
            value.gi_shader=base+"/pcloud_gi.comp.spv";
            value.lighting_shader=base+"/pcloud_light.comp.spv";
            value.tonemap_shader=base+"/pcloud_tone.comp.spv";
            value.fullscreen_shader=base+"/pcloud_fullscreen.vert.spv";
            value.presentation_shader=base+"/pcloud_present.frag.spv";
            return value;
        }
    };
    struct targets {vk::Image grid{};vk::Buffer points{};vk::Image hdr{},ldr{};};
    using overlay_callback=std::function<void(vk::CommandBuffer,vk::Extent2D)>;
    // Explicit targets work with any VulkanWrapper-based renderer.
    void create(const configuration&,const std::vector<gbuffer>&,
                const presentation_target&);
    // Convenience for existing VulkanWrapper renderers with an active
    // vk::swapchain. No separate presentation_target boilerplate required.
    void create(const std::vector<gbuffer>&,
                const configuration& shaders={});
    void set_scene(std::shared_ptr<const ::vs::scene::data>);
    // Optional user-authored outdoor GI coverage, independent of local
    // point-light radius. The atmosphere itself is supplied later.
    // Call synchronize_scene() after changing coverage at an idle boundary.
    void set_probe_coverage(std::optional<::vs::scene::collision_box> bounds);
    void synchronize_scene();
    struct shadow_sphere {
        vec3 center{};
        float radius=0.f;
    };
    // CPU-only BVH refit can run before swapchain acquisition; GPU uploads
    // remain indexed by the acquired frame, so in-flight buffers are safe.
    void prepare_dynamic_geometry(const std::vector<triangle>& triangles,
                                  const std::vector<shadow_sphere>& spheres={});
    void upload_dynamic_geometry(std::uint32_t image);
    // Compatibility one-call path for existing clients.
    void update_dynamic_geometry(std::uint32_t image,
                                 const std::vector<triangle>& triangles,
                                 const std::vector<shadow_sphere>& spheres={});
    // Update light color/intensity without rebuilding unchanged scene geometry.
    // Position, radius, active mask and light count must be unchanged.
    void update_lights(const std::vector<light>&);
    // Invalidate only GI cells affected by a lighting/material change.
    // Geometry replacement must still use set_scene/synchronize_scene until
    // the static GPU BVH supports independent region pages.
    void invalidate_region(const ::vs::scene::collision_box& bounds);
    // Region-local opaque shadows are built on CPU and uploaded lazily for
    // the acquired swapchain image. World generation is not part of PCloud.
    bool replace_shadow_region(::vs::spatial::region_id id,
        std::uint64_t revision,std::span<const triangle> faces);
    bool remove_shadow_region(::vs::spatial::region_id id,
        std::uint64_t revision);
    bool forget_shadow_region(::vs::spatial::region_id id);
    void clear_shadow_regions();
    // Region-scoped radiance/probe invalidation without static BVH rebuild.
    // Caller waits for in-flight frames before updating CPU-visible buffers.
    // Use set_scene() when triangles or probe topology have changed.
    void invalidate_radiance(const collision_box& affected);
    // Rebuild only colored transmission when glass tint/pose changes.
    // Preserves the opaque BVH, GI probe positions and resource topology.
    // Caller guarantees that submitted GPU work is idle.
    void update_transmission(const ::vs::scene::data&);
    // Borrow existing exact opaque shadow acceleration for other VS renderers.
    // GPU buffers remain owned by PCloud and must be rebound after scene sync.
    struct shadow_geometry {
        vk::Buffer triangles{},nodes{},light_ranges{},transmission{};
        vk::Buffer glass_vertices{},glass_nodes{},glass_filters{},light_shapes{};
    };
    shadow_geometry opaque_shadows();
    struct inspection_resources {
        vk::ImageView grid{};
        vk::Buffer points{},nodes{};
        vk::Extent3D extent{};
        std::array<float,4> origin_spacing{};
        std::uint32_t point_count=0,node_count=0,shared_nodes=0;
    };
    // Borrow the exact live data; no CPU readback or alternate simulation.
    inspection_resources inspect(std::uint32_t image);
    // Render the same opaque PCloud lighting from a mirrored raster camera.
    // Uses a second full-resolution G-buffer, existing probe field and shadow
    // BVH, and a separate per-image HDR/LDR target. No ray-traced primary hit.
    void create_reflection(const std::vector<gbuffer>&);
    void reflection_lighting(vk::CommandBuffer,std::uint32_t,const view_basis&);
    void reflection_tonemap(vk::CommandBuffer,std::uint32_t);
    vk::ImageView reflection_ldr_view(std::uint32_t);
    targets reflection_images(std::uint32_t);

    // Per-swapchain-image GI build state. A complete grid may still contain
    // zero radiance when no light reaches the probe coverage volume.
    struct gi_progress {
        std::uint32_t points=0;
        std::uint32_t completed_groups=0,total_groups=0;
        bool pending=false,initialized=false;
    };
    gi_progress progress(std::uint32_t image);
    struct lighting_statistics {
        std::uint32_t lights=0;
        std::uint32_t dynamic_shadow_lights=0;
        std::uint32_t pixel_rate=1;
        std::uint32_t dynamic_primitives=0;
    };
    lighting_statistics lighting_stats(std::uint32_t image);
    // Call only after the frame command buffer was successfully submitted.
    void updated(std::uint32_t);
    void set_exposure(float);
    targets images(std::uint32_t);
    vk::ImageView ldr_view(std::uint32_t);
    vk::ImageView hdr_view(std::uint32_t);
    vk::ImageView probe_view(std::uint32_t);
    std::array<float,4> probe_space();
    vk::RenderPass presentation_pass();
    void cloud(vk::CommandBuffer,std::uint32_t);
    void gi(vk::CommandBuffer,std::uint32_t);
    void lighting(vk::CommandBuffer,std::uint32_t,const camera_frame&);
    void tonemap(vk::CommandBuffer,std::uint32_t,const camera_frame&);
    void present(vk::CommandBuffer,std::uint32_t,const overlay_callback& overlay={});
    // One-call path: records cloud + GI + lighting + tonemap + present with
    // internally tracked Vulkan memory/image barriers for this swapchain image.
    // G-buffer must be finished and shader-readable before record().
    void record(vk::CommandBuffer,std::uint32_t,const camera_frame&,
                const overlay_callback& overlay={});
    // Call only after successful queue submission, not merely after record().
    void release();
}
