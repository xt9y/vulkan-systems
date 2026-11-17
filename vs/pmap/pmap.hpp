#pragma once
#include <vk/vk.hpp>
#include <vs/scene/scene.hpp>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace vs::pmap
{
    struct configuration
    {
        std::string shader_directory="build/shaders";
        std::uint32_t photons=8192,bounces=4;
        std::uint32_t gi_updates=64,direct_updates=16;
        std::uint32_t direct_spp=4,atlas_tile=64;
        std::uint32_t denoise_passes=1,edge_spp=2,screen_spp=2;
        float photon_radius=.30f;
        bool screen_shadows=true;
        bool transport_only=false; // avoids fullscreen work and large atlases
    };
    struct progress
    {
        std::uint32_t gi_updates=0,gi_target=0;
        std::uint32_t direct_updates=0,direct_target=0;
        std::uint64_t submitted_frames=0;
    };
    struct photon_output {
        vk::Buffer sorted{},used{};
        std::uint32_t capacity=0;
    };
    // Read-only GPU buffers for generic visibility/radiance consumers.
    // Exactly the same BVH and face data used by photon tracing.
    struct geometry_output {
        vk::Buffer faces{}, nodes{}, triangles{}, scene_info{};
        vk::Buffer photons{}, offsets{},face_ids{};
    };
    struct geometry_update {
        std::uint32_t face=0;
        ::vs::scene::surface value{};
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
        void create(const ::vs::scene::data&,vk::Extent2D,const configuration& config={});
        void resize(vk::Extent2D);
        void set_scene(const ::vs::scene::data&);
        // Update without rebuilding GPU pipelines. The host must not
        // change GPU-visible data while its prior submission is in flight.
        void set_glass(const ::vs::scene::glass_sphere&);
        void set_emission(std::uint32_t surface_index,::vs::scene::vec3 emission);
        // Reset convergence and camera history without recreating pipelines.
        void invalidate();
        // Records compute only, into the host's active command buffer.
        // Host submits this buffer exactly once and waits for completion
        // before recording another frame through the same mapper.
        image_output record(vk::CommandBuffer,const ::vs::scene::camera&);
        // Transport-only: emission, BVH interactions, compact spatial photons.
        // No camera rays, full-screen gathering, or tonemapping.
        photon_output record_transport(vk::CommandBuffer);
        photon_output photon_buffers() const;
        geometry_output geometry_buffers() const;
        // Scene topology stays fixed. Refit world-space surfaces with
        // skinned animation in independent per-swapchain-image buffers.
        // Call after acquiring the image and waiting for its fence.
        geometry_output refit_geometry(std::uint32_t image,
                                      const std::vector<geometry_update>&);
        image_output output() const;
        image_output surface_positions() const;
        progress status() const;
        const ::vs::scene::data& current_scene() const;
        void destroy();

    private:
        struct implementation;
        std::unique_ptr<implementation> state_;
    };
}
