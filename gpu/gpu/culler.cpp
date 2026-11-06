#include "culler.hpp"
#include "transfer.hpp"
#include <cstddef>
#include <stdexcept>

namespace gpu {
    void culler::create(const std::string& spirv,vk::DescriptorSetLayout set) {
        if(!set)throw std::invalid_argument("culler requires descriptor layout");
        kernel_.create(spirv,{set},sizeof(parameters));
    }
    void culler::begin(vk::CommandBuffer command,const std::vector<vk::Buffer>& arguments) {
        if(!command)throw std::invalid_argument("culler requires command buffer");
        for(const auto buffer:arguments)
            transfer::fill(command,buffer,
                offsetof(vk::DrawIndexedIndirectCommand,instanceCount),
                sizeof(std::uint32_t),0);
        if(arguments.empty())return;
        vk::MemoryBarrier barrier{};
        barrier.srcAccessMask=vk::AccessFlagBits::eTransferWrite;
        barrier.dstAccessMask=vk::AccessFlagBits::eShaderRead|
                              vk::AccessFlagBits::eShaderWrite;
        command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
            vk::PipelineStageFlagBits::eComputeShader,{}, {barrier}, {}, {});
    }
    void culler::dispatch(vk::CommandBuffer command,vk::DescriptorSet set,
        const std::array<vk::util::math::vec4,6>& planes,
        std::uint32_t count,std::uint32_t indices) const {
        if(!count)return;
        if(!set || !indices)throw std::invalid_argument("invalid culler mesh");
        parameters data{}; data.planes=planes; data.count=count; data.indices=indices;
        kernel_.dispatch(command,(count+127u)/128u,1,1,{set},&data,sizeof(data));
    }
    void culler::end(vk::CommandBuffer command) const {
        if(!command)throw std::invalid_argument("culler requires command buffer");
        vk::MemoryBarrier barrier{};
        barrier.srcAccessMask=vk::AccessFlagBits::eShaderWrite;
        barrier.dstAccessMask=vk::AccessFlagBits::eVertexAttributeRead|
                              vk::AccessFlagBits::eIndirectCommandRead;
        command.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
            vk::PipelineStageFlagBits::eVertexInput|
            vk::PipelineStageFlagBits::eDrawIndirect,{}, {barrier}, {}, {});
    }
}
