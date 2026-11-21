#pragma once
#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <vector>
#include <vk/util/math/vector.hpp>
#include <vs/gpu/compute.hpp>

namespace vs::gpu
{
// std140 view input: one copy for primary and one for reflected culling.
struct alignas(16) virtual_view
{
    std::array<vk::util::math::vec4, 6> planes{};
    std::array<float, 16> previous_vp{};
    std::array<float, 4> previous_camera{};                // xyz, previous pixel focal length
    std::array<float, 4> options{};                        // enabled, width, height, mip count
    std::array<float, 4> animated_count{};                 // count of conservative blockers
    std::array<std::array<float, 4>, 64> animated_rects{}; // xy min/max in UV space
};

static_assert(sizeof(virtual_view) == 1232);
// Set-0 binding 10 page descriptors, independent of the renderer's material
// and vertex formats. Index counts come from each resident meshlet page.
struct alignas(16) virtual_page_meta
{
    std::array<float, 4> sphere{};       // Local-space conservative page bound
    std::array<std::uint32_t, 4> info{}; // x: level, y: index count
};

static_assert(sizeof(virtual_page_meta) == 32);
class virtual_culler
{
public:
    struct draw_buffer
    {
        vk::Buffer handle{};
        std::uint32_t levels = 1;
    };

    void create(const std::string &shader, vk::DescriptorSetLayout layout,
                vk::DescriptorSetLayout view_layout);
    void reset() { kernel_.reset(); }
    void begin(vk::CommandBuffer, const std::vector<draw_buffer> &);
    void dispatch(vk::CommandBuffer, vk::DescriptorSet,
                  vk::DescriptorSet view_set,
                  vk::util::math::vec3 eye, float pixel_scale,
                  std::uint32_t instances,
                  std::span<const std::uint32_t> indices,
                  std::span<const float> errors,
                  std::uint32_t resident_mask = 15u,
                  std::uint32_t page_count = 0u,
                  bool allow_occlusion = true) const;
    void end(vk::CommandBuffer) const;

private:
    struct parameters
    {
        std::array<float, 4> camera{}, errors{};
        std::array<std::uint32_t, 4> info{}, indices{};
    };
    ::vs::gpu::compute kernel_;
    // 64-byte per-mesh push constants; frustum and previous-frame Hi-Z
    // state live in an immutable descriptor per recorded camera pass.
};
} // namespace vs::gpu
