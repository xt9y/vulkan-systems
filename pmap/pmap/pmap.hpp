#pragma once
#include <vk/vk.hpp>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace pm
{
    using vec3=vk::util::math::vec3;

    // World-space diffuse, emissive or zero-albedo receiver. A quad
    // spans origin + [0,1]*u + [0,1]*v. A triangle spans barycentric
    // origin + a*u + b*v with a>=0, b>=0, a+b<=1.
    struct surface
    {
        vec3 origin{},u{},v{},normal{};
        vec3 albedo{.725f,.710f,.680f};
        vec3 emission{};
        bool triangle=false;
    };
    // Mark receiver texels hidden behind another surface, e.g. a floor
    // covered by a box top, using projection along receiver normal.
    struct cutout {std::uint32_t receiver=0,cover=0;};
    struct glass_sphere
    {
        vec3 center{};
        float radius=0.f;
        float ior=1.52f;
    };
    struct scene
    {
        std::vector<surface> surfaces;
        std::uint32_t emitter=0;
        std::vector<cutout> cutouts;
        glass_sphere glass{};
    };
    struct view
    {
        vec3 eye{},right{1,0,0},up{0,1,0};
        float vertical_fov=0.68f; // radians
    };
    struct configuration
    {
        std::string shader_directory="build/shaders";
        std::uint32_t photons=8192,bounces=4;
        std::uint32_t gi_updates=64,direct_updates=16;
        std::uint32_t direct_spp=4,atlas_tile=64;
        std::uint32_t denoise_passes=1,edge_spp=2,screen_spp=2,debug_view=0;
        float photon_radius=.30f;
        bool screen_shadows=true;
    };
    struct progress
    {
        std::uint32_t gi_updates=0,gi_target=0;
        std::uint32_t direct_updates=0,direct_target=0;
        std::uint64_t submitted_frames=0;
    };
    struct image_output
    {
        vk::Image image{};
        vk::ImageView view{};
        vk::Extent2D extent{};
        vk::ImageLayout layout=vk::ImageLayout::eGeneral;
    };

    // VulkanWrapper must already have an initialized device/queue.
    // This class owns photon maps, acceleration data, temporal history,
    // direct visibility moments, GPU pipelines and all internal barriers.
    // It never creates a swapchain, GLFW window, command pool, or renderer.
    class mapper
    {
    public:
        mapper();
        ~mapper();
        mapper(const mapper&)=delete;
        mapper& operator=(const mapper&)=delete;

        // Caller waits for GPU idle before create/resize/set_scene/destroy.
        void create(const scene&,vk::Extent2D,const configuration& config={});
        void resize(vk::Extent2D);
        void set_scene(const scene&);
        // Update without rebuilding GPU pipelines. The host must not
        // change GPU-visible data while its prior submission is in flight.
        void set_glass(const glass_sphere&);
        void set_emission(std::uint32_t surface_index,vec3 emission);
        // Reset convergence and camera history without recreating pipelines.
        void invalidate();
        // Records compute only, into the host's active command buffer.
        // Host submits this buffer exactly once and waits for completion
        // before recording another frame through the same mapper.
        image_output record(vk::CommandBuffer,const view&);
        image_output output() const;
        progress status() const;
        const scene& current_scene() const;
        void destroy();

    private:
        struct implementation;
        std::unique_ptr<implementation> state_;
    };
}
