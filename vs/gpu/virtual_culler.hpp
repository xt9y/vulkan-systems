#pragma once
#include <vs/gpu/compute.hpp>
#include <vk/util/math/vector.hpp>
#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace vs::gpu {
class virtual_culler {
public:
    struct draw_buffer {vk::Buffer handle{};std::uint32_t levels=1;};
    void create(const std::string& shader,vk::DescriptorSetLayout layout);
    void reset(){kernel_.reset();}
    void begin(vk::CommandBuffer,const std::vector<draw_buffer>&);
    void dispatch(vk::CommandBuffer,vk::DescriptorSet,
                  const std::array<vk::util::math::vec4,6>& planes,
                  vk::util::math::vec3 eye,float pixel_scale,
                  std::uint32_t instances,
                  std::span<const std::uint32_t> indices,
                  std::span<const float> errors) const;
    void end(vk::CommandBuffer) const;
private:
    struct parameters {
        std::array<vk::util::math::vec4,6> planes{};
        std::array<float,4> camera{},errors{};
        std::array<std::uint32_t,4> info{},indices{};
    };
    ::vs::gpu::compute kernel_;
};
}
