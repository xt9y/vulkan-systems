#pragma once
#include <vk/vk.hpp>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace pt
{
    using vec3=vk::util::math::vec3;

    // Quads use origin + a*u + b*v (0..1). Triangles use a+b<=1.
    // Normals specify the front face; the integrator is one-sided.
    struct surface
    {
        vec3 origin{},u{},v{},normal{};
        vec3 albedo{.725f,.710f,.680f};
        vec3 emission{};
        bool triangle=false;
    };
    struct scene
    {
        std::vector<surface> surfaces;
        std::uint32_t emitter=0; // Single sampled area-light surface.
    };
    struct view
    {
        vec3 eye{},right{1.f,0.f,0.f},up{0.f,1.f,0.f};
        float vertical_fov=0.680f; // Radians; view axes are unit vectors.
    };
    struct configuration
    {
        std::string shader_directory="build/shaders";
        std::uint32_t bounces=2;
        std::uint32_t light_spp=2; // 1, 2, 4
        std::uint32_t denoise_radius=1; // 0..2
        bool temporal=true;
    };
    struct image_output
    {
        vk::Image image{};
        vk::ImageView view{};
        vk::Extent2D extent{};
        vk::ImageLayout layout=vk::ImageLayout::eGeneral;
    };
    struct progress
    {
        std::uint64_t frames=0;
        std::uint32_t stable_frames=0;
        bool moving=false;
    };

    // Uses the host's VulkanWrapper device and its already-recording
    // command buffer. No swapchain, presentation, GLFW or global context.
    // One frame may be in flight; wait for completion before reusing tracer.
    class tracer
    {
    public:
        tracer();
        ~tracer();
        tracer(const tracer&)=delete;
        tracer& operator=(const tracer&)=delete;

        // Device must be idle when creating, resizing or replacing the scene.
        void create(const scene&,vk::Extent2D,const configuration&={});
        void resize(vk::Extent2D);
        void set_scene(const scene&);
        void invalidate();
        image_output record(vk::CommandBuffer,const view&);
        image_output output() const; // Display-only denoised HDR, FP16.
        image_output surface_positions() const; // First hits + surface ID.
        progress status() const;
        void destroy();

    private:
        struct implementation;
        std::unique_ptr<implementation> state_;
    };
}
