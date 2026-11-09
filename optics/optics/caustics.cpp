#include "caustics.hpp"
#include <array>
#include <stdexcept>

namespace optics {
void caustics::create(vk::Extent2D extent,const std::vector<vk::ImageView>& depth,
                      const char* shader)
{
    if(!extent.width || !extent.height || depth.empty() || !shader)
        throw std::invalid_argument("optics: invalid caustic target");
    reset();
    try {
        extent_=extent;
        outputs_.resize(depth.size());
        const vk::DeviceSize bytes=vk::DeviceSize(extent.width)*extent.height*3u*sizeof(std::uint32_t);
        for(auto& out:outputs_) {
            out=std::make_unique<gpu::buffer>();
            out->create(bytes,vk::BufferUsageFlagBits::eStorageBuffer |
                              vk::BufferUsageFlagBits::eTransferDst,
                        vk::MemoryPropertyFlagBits::eDeviceLocal);
        }
        std::vector<vk::DescriptorSetLayoutBinding> bindings(4);
        for(std::uint32_t i=0;i<4;++i) {
            bindings[i].binding=i;
            bindings[i].descriptorCount=1;
            bindings[i].descriptorType=i==3 ?
                vk::DescriptorType::eCombinedImageSampler :
                vk::DescriptorType::eStorageBuffer;
            bindings[i].stageFlags=vk::ShaderStageFlagBits::eCompute;
        }
        sets_.create(bindings,static_cast<std::uint32_t>(depth.size()));
        vk::SamplerCreateInfo sampler{};
        sampler.magFilter=vk::Filter::eNearest;
        sampler.minFilter=vk::Filter::eNearest;
        sampler.mipmapMode=vk::SamplerMipmapMode::eNearest;
        sampler.addressModeU=sampler.addressModeV=sampler.addressModeW=
            vk::SamplerAddressMode::eClampToEdge;
        sampler_=vk::device::get()->handle.createSampler(sampler);
        for(std::uint32_t i=0;i<depth.size();++i) {
            vk::DescriptorImageInfo image{};
            image.imageView=depth[i];
            image.sampler=sampler_;
            image.imageLayout=vk::ImageLayout::eDepthStencilReadOnlyOptimal;
            sets_.write_image(i,3,image);
            sets_.write_buffer(i,2,outputs_[i]->descriptor());
        }
        program_.create(shader,{sets_.layout()},sizeof(vk::util::math::mat4)+sizeof(std::array<std::uint32_t,4>));
    }catch(...) {reset();throw;}
}
void caustics::set_photons(photons source)
{
    if(!source.capacity) {
        source_={};
        return;
    }
    if(!source.sorted || !source.used)
        throw std::invalid_argument("optics: missing photon transport buffers");
    source_=source;
    for(std::uint32_t i=0;i<outputs_.size();++i) {
        vk::DescriptorBufferInfo sorted{source.sorted,0,VK_WHOLE_SIZE};
        vk::DescriptorBufferInfo count{source.used,0,sizeof(std::uint32_t)};
        sets_.write_buffer(i,0,sorted);
        sets_.write_buffer(i,1,count);
    }
}
void caustics::record(vk::CommandBuffer cmd,std::uint32_t image,
                      const vk::util::math::mat4& vp)
{
    if(!cmd || image>=outputs_.size())throw std::invalid_argument("optics: invalid image");
    auto& out=*outputs_[image];
    cmd.fillBuffer(out.handle(),0,out.size(),0);
    vk::BufferMemoryBarrier transfer{};
    transfer.srcAccessMask=vk::AccessFlagBits::eTransferWrite;
    transfer.dstAccessMask=vk::AccessFlagBits::eShaderRead |
                           vk::AccessFlagBits::eShaderWrite;
    transfer.srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
    transfer.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
    transfer.buffer=out.handle();
    transfer.size=VK_WHOLE_SIZE;
    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                        vk::PipelineStageFlagBits::eComputeShader,{},
                        {},{transfer},{});
    if(source_.capacity) {
        struct push {vk::util::math::mat4 vp;std::array<std::uint32_t,4> args;};
        const push params{vp,{extent_.width,extent_.height,source_.capacity,0}};
        program_.dispatch(cmd,(source_.capacity+63u)/64u,1u,1u,
                          {sets_.set(image)},&params,sizeof(params));
    }
    vk::BufferMemoryBarrier after=transfer;
    after.srcAccessMask=vk::AccessFlagBits::eShaderWrite |
                        vk::AccessFlagBits::eTransferWrite;
    after.dstAccessMask=vk::AccessFlagBits::eShaderRead;
    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader |
                        vk::PipelineStageFlagBits::eTransfer,
                        vk::PipelineStageFlagBits::eFragmentShader,{},
                        {},{after},{});
}
vk::Buffer caustics::buffer(std::uint32_t image) const
{
    return outputs_.at(image)->handle();
}
void caustics::reset() noexcept
{
    program_.reset();
    sets_.reset();
    outputs_.clear();
    if(sampler_)vk::device::get()->handle.destroySampler(sampler_);
    sampler_=nullptr;
    source_={};
    extent_=vk::Extent2D{};
}
}
