#pragma once
#include "transport.hpp"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace transport::hierarchy
{
    // The local tangent vectors are the full edges of a single finite patch.
    // The original transport::patch layout remains compatible with renderers.
    struct surface
    {
        transport::patch value{};
        vec4 tangent_u{},tangent_v{};
    };

    struct options
    {
        float far_field_ratio=0.30f;
        float normal_threshold=0.995f;
        std::uint32_t max_edges=3000000;
        std::uint32_t initial_iterations=32;
    };

    struct statistics
    {
        std::uint32_t surface_count=0,node_count=0,edge_count=0;
        std::uint32_t reciprocal_pairs=0;
        float initial_max_row=0.f,global_scale=1.f,graph_build_ms=0.f;
        float residual=0.f,reciprocity_max_error=0.f;
        float operator_norm_upper=0.f,discrete_error_bound=0.f;
    };

    class solver
    {
    public:
        struct state; // Opaque implementation data, not part of the GPU API.
        solver();
        ~solver();
        solver(const solver&)=delete;
        solver& operator=(const solver&)=delete;

        // An initialized VulkanEngine device is required.
        // The optional oriented boxes are a limited geometric occluder backend.
        void init(const std::vector<surface>& surfaces,
                  const std::vector<transport::oriented_blocker>& occluders,
                  const std::string& shader_prefix,options config={});
        void record_initial(vk::CommandBuffer command,std::uint32_t iterations);
        // Continue from the current radiance[0] solution without clearing it.
        void record_refine(vk::CommandBuffer command,std::uint32_t iterations);
        // Call from a host thread after the previous GPU use is complete.
        void update_emission(std::uint32_t index,vec4 emission);
        // Incremental delta-emission solution; static geometry only.
        void record_delta(vk::CommandBuffer command,std::uint32_t bounces);
        void record_residual(vk::CommandBuffer command);
        // Only after the recorded GPU submission is complete.
        float residual() const;
        // Certified upper bound for the *discrete* linear transport system,
        // not for spatial/visibility quadrature errors.
        float error_bound() const;
        const statistics& metrics() const;
        vk::DescriptorBufferInfo patches_descriptor() const;
        vk::DescriptorBufferInfo radiance_descriptor() const;
        vk::DescriptorBufferInfo direct_descriptor() const;
        vk::DescriptorBufferInfo oriented_descriptor() const;
        void shutdown();

    private:
        std::unique_ptr<state> state_;
    };
}
