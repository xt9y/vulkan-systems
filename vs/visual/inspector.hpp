#pragma once
#include <vs/gpu/descriptors.hpp>
#include <vs/gpu/pipeline.hpp>
#include <vs/render/pipeline.hpp>
#include <vk/vk.hpp>
#include <array>
#include <cstdint>
#include <vector>

namespace vs::visual
{
    // The buffers belong to PCloud; the inspector only borrows them.
    struct source
    {
        vk::ImageView probes{};
        vk::Buffer point_cloud{},bvh{};
        vk::Extent3D extent{};
        std::array<float,4> grid{}; // world origin, lattice spacing
        std::uint32_t point_count=0,node_count=0,shared_nodes=0;
    };
    class inspector
    {
    public:
        ~inspector(){reset();}
        inspector(const inspector&)=delete;
        inspector& operator=(const inspector&)=delete;
        inspector()=default;
        void create(vk::RenderPass,std::uint32_t frames);
        // Update only when previously submitted frames have completed.
        void bind(const std::vector<source>& per_frame);
        // Mode 1 = BVH, 2 = active probes, 3 = actual surface point cloud.
        // Projection must be column-major, as consumed by GLSL mat4.
        void record(vk::CommandBuffer,std::uint32_t image,std::uint32_t mode,
                    const std::array<float,16>& projection,
                    vk::Extent2D viewport) const;
        void reset() noexcept;
    private:
        gpu::descriptors sets_;
        gpu::pipeline_layout layout_;
        render::pipeline line_,disc_;
        vk::Sampler sampler_{};
        std::vector<source> sources_;
        std::uint32_t frames_=0;
    };
}
