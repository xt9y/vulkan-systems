#pragma once
#include <vk/vk.hpp>
#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace transport
{
    using vec4=vk::util::math::vec4;

    // std430 compatible. Positions are patch centers, w is their area.
    // The exchange solver models Lambertian reflection only.
    struct alignas(16) patch
    {
        vec4 position_area{};
        vec4 normal{};
        vec4 albedo{};
        vec4 emission{};
    };
    static_assert(sizeof(patch)==64);

    struct alignas(16) blocker
    {
        vec4 minimum{},maximum{};
    };
    static_assert(sizeof(blocker)==32);

    // A reusable oriented bounding box, with orthonormal world axes.
    // half_extent.xyz stores extent along each corresponding axis.
    struct alignas(16) oriented_blocker
    {
        vec4 center{},half_extent{},axis_x{},axis_y{},axis_z{};
    };
    static_assert(sizeof(oriented_blocker)==80);

    // Shader order: exchange construction, Jacobi solve, delta injection, delta propagation.
    using shader_paths=std::array<std::string,5>;

    struct state
    {
        vk::buffer::resource patches,blockers,oriented_blockers,exchange;
        std::array<vk::buffer::resource,2> radiance,delta;
        vk::buffer::resource previous_emission,direct;
        vk::DescriptorSetLayout layout{};
        vk::DescriptorPool pool{};
        std::array<vk::DescriptorSet,4> sets{};
        vk::PipelineLayout pipeline_layout{};
        std::array<vk::Pipeline,5> pipelines{};
        std::uint32_t patch_count=0,blocker_count=0,oriented_count=0;
    };

    state* get();
    void init(const std::vector<patch>& patches,
              const std::vector<blocker>& blockers,
              const shader_paths& shaders,
              const std::vector<oriented_blocker>& oriented={});
    // Build is required when geometry, visibility or patch areas change.
    // Records GPU writes and all necessary internal compute barriers.
    void record_build(vk::CommandBuffer command);
    // Computes the first-bounce component to replace its low-resolution
    // approximation with high-quality per-pixel area-light sampling.
    void record_direct(vk::CommandBuffer command);
    // Even iteration count keeps the public output bound to radiance[0].
    // The scene is stable between calls; callers synchronize host updates.
    void record_solve(vk::CommandBuffer command,std::uint32_t iterations);
    // Neumann delta update: retain previous lighting and only propagate changes
    // to emission. Geometry/visibility must remain unchanged.
    void record_delta(vk::CommandBuffer command,std::uint32_t terms);
    void update_emission(std::uint32_t index,vec4 emission);
    vk::DescriptorBufferInfo patches_descriptor();
    vk::DescriptorBufferInfo radiance_descriptor();
    vk::DescriptorBufferInfo direct_descriptor();
    vk::DescriptorBufferInfo oriented_descriptor();
    void shutdown();
}
