#include <vs/gpu/virtual_culler.hpp>
#include <vs/gpu/transfer.hpp>
#include <algorithm>
#include <cstddef>
#include <stdexcept>

namespace vs::gpu {
void virtual_culler::create(const std::string& shader,
                            vk::DescriptorSetLayout layout,
                            vk::DescriptorSetLayout view_layout)
{
    if(!layout || !view_layout)
        throw std::invalid_argument("Missing virtual culler layouts");
    kernel_.create(shader,{layout,view_layout},sizeof(parameters));
}
void virtual_culler::begin(vk::CommandBuffer cmd,const std::vector<draw_buffer>& buffers)
{
    if(!cmd)throw std::invalid_argument("Invalid culler command");
    for(const auto& buffer:buffers) {
        if(!buffer.handle || !buffer.levels || buffer.levels>4u)
            throw std::invalid_argument("Invalid virtual culler buffer");
        ::vs::gpu::transfer::fill(cmd,buffer.handle,0u,
            buffer.levels*sizeof(vk::DrawIndexedIndirectCommand),0u);
    }
    if(buffers.empty())return;
    vk::MemoryBarrier barrier{};
    barrier.srcAccessMask=vk::AccessFlagBits::eTransferWrite;
    barrier.dstAccessMask=vk::AccessFlagBits::eShaderRead|
                          vk::AccessFlagBits::eShaderWrite;
    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
        vk::PipelineStageFlagBits::eComputeShader,{}, {barrier},{},{});
}
void virtual_culler::dispatch(vk::CommandBuffer cmd,vk::DescriptorSet set,
    vk::DescriptorSet view_set,
    vk::util::math::vec3 eye,float pixel_scale,std::uint32_t instances,
    std::span<const std::uint32_t> indices,
    std::span<const float> errors,
    std::uint32_t resident_mask) const
{
    if(!instances)return;
    if(!cmd || !set || !view_set || indices.empty() || indices.size()>4u ||
       errors.size()!=indices.size())
        throw std::invalid_argument("Invalid virtual culler dispatch");
    parameters push{};
    push.camera={eye.x,eye.y,eye.z,pixel_scale};
    push.info={instances,std::uint32_t(indices.size()),0u,resident_mask};
    for(std::size_t i=0;i<indices.size();++i) {
        push.indices[i]=indices[i];
        push.errors[i]=std::max(0.f,errors[i]);
    }
    kernel_.dispatch(cmd,(instances+127u)/128u,1u,1u,
        {set,view_set},&push,sizeof(push));
}
void virtual_culler::end(vk::CommandBuffer cmd) const
{
    if(!cmd)throw std::invalid_argument("Invalid culler command");
    vk::MemoryBarrier barrier{};
    barrier.srcAccessMask=vk::AccessFlagBits::eShaderWrite;
    barrier.dstAccessMask=vk::AccessFlagBits::eVertexAttributeRead |
                          vk::AccessFlagBits::eIndirectCommandRead;
    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eVertexInput |
        vk::PipelineStageFlagBits::eDrawIndirect,{}, {barrier},{},{});
}
}
