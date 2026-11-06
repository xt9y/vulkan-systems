#pragma once
#include "compute.hpp"
#include <vk/util/math/vector.hpp>
#include <array>
#include <string>
#include <vector>

namespace gpu {
    // Frustum sphere culling, compact visible instance IDs and indexed indirect
    // arguments. Shader set bindings: 0 transforms, 1 vec4 spheres,
    // 2 writable uint visible IDs, 3 writable VkDrawIndexedIndirectCommand.
    class culler {
    public:
        void create(const std::string& spirv, vk::DescriptorSetLayout set);
        void reset() { kernel_.reset(); }
        void begin(vk::CommandBuffer cmd,const std::vector<vk::Buffer>& indirect);
        void dispatch(vk::CommandBuffer cmd,vk::DescriptorSet set,
            const std::array<vk::util::math::vec4,6>& planes,
            std::uint32_t instance_count,std::uint32_t index_count) const;
        void end(vk::CommandBuffer cmd) const;
    private:
        struct parameters {
            std::array<vk::util::math::vec4,6> planes{};
            std::uint32_t count=0;
            std::uint32_t indices=0;
        };
        gpu::compute kernel_;
    };
}
