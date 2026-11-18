#pragma once
#include <vs/gpu/compute.hpp>
#include <vk/util/math/vector.hpp>
#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace vs::gpu {
// std140 view input: one copy for primary and one for reflected culling.
struct alignas(16) virtual_view {
    std::array<vk::util::math::vec4,6> planes{};
    std::array<float,16> previous_vp{};
    std::array<float,4> previous_camera{}; // xyz, previous pixel focal length
    std::array<float,4> options{};         // enabled, width, height, mip count
};
static_assert(sizeof(virtual_view)==192);
class virtual_culler {
public:
    struct draw_buffer {vk::Buffer handle{};std::uint32_t levels=1;};
    void create(const std::string& shader,vk::DescriptorSetLayout layout,
                vk::DescriptorSetLayout view_layout);
    void reset(){kernel_.reset();}
    void begin(vk::CommandBuffer,const std::vector<draw_buffer>&);
    void dispatch(vk::CommandBuffer,vk::DescriptorSet,
                  vk::DescriptorSet view_set,
                  vk::util::math::vec3 eye,float pixel_scale,
                  std::uint32_t instances,
                  std::span<const std::uint32_t> indices,
                  std::span<const float> errors,
                  std::uint32_t resident_mask=15u) const;
    void end(vk::CommandBuffer) const;
private:
    struct parameters {
        std::array<float,4> camera{},errors{};
        std::array<std::uint32_t,4> info{},indices{};
    };
    ::vs::gpu::compute kernel_;
    // 64-byte per-mesh push constants; frustum and previous-frame Hi-Z
    // state live in an immutable descriptor per recorded camera pass.
};
}
