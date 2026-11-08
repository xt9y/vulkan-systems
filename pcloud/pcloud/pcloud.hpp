#pragma once
#include <vk/vk.hpp>
#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>
namespace pc
{
    using vec3=vk::util::math::vec3;
    struct light {vec3 position{};float radius=1.f;vec3 color{1.f,1.f,1.f};float strength=1.f;};
    struct collision_box {vec3 minimum{},maximum{};};
    struct triangle {vec3 a{},b{},c{};};

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
            if(length_squared(near-light.position)<=light.radius*light.radius)
                return true;
        }
        return false;
    }

    struct world
    {
        std::vector<light> lights;
        std::vector<collision_box> collision_boxes;
        std::vector<triangle> triangles;
    };
    struct camera_frame {vec3 eye{};float yaw=0.f,pitch=0.f,fov=1.0471976f;};
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
    };
    struct targets {vk::Image grid{};vk::Buffer points{};vk::Image hdr{},ldr{};};
    using overlay_callback=std::function<void(vk::CommandBuffer,vk::Extent2D)>;
    void create(const configuration&,const std::vector<gbuffer>&,
                const presentation_target&);
    void set_scene(std::shared_ptr<const world>);
    void synchronize_scene();
    void updated(std::uint32_t);
    void set_exposure(float);
    void toggle_probes();
    targets images(std::uint32_t);
    void cloud(vk::CommandBuffer,std::uint32_t);
    void gi(vk::CommandBuffer,std::uint32_t);
    void lighting(vk::CommandBuffer,std::uint32_t,const camera_frame&);
    void tonemap(vk::CommandBuffer,std::uint32_t,const camera_frame&);
    void present(vk::CommandBuffer,std::uint32_t,const overlay_callback& overlay={});
    void release();
}
