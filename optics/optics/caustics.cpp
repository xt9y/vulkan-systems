#include "caustics.hpp"
#include <array>
#include <algorithm>
#include <cstring>
#include <utility>
#include <stdexcept>

namespace optics {
void caustics::create(vk::Extent2D extent,const std::vector<vk::ImageView>& depth,
                      const std::vector<vk::ImageView>& hdr,
                      const char* shader,const char* resolve)
{
    if(!extent.width || !extent.height || depth.empty() ||
       depth.size()!=hdr.size() || !shader || !resolve)
        throw std::invalid_argument("optics: invalid caustic target");
    reset();
    try {
        extent_=extent;
        outputs_.resize(depth.size());
        cached_views_.resize(depth.size());
        valid_views_.assign(depth.size(),0u);
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
        std::vector<vk::DescriptorSetLayoutBinding> resolve_bindings(2);
        resolve_bindings[0].binding=0;
        resolve_bindings[0].descriptorType=vk::DescriptorType::eStorageBuffer;
        resolve_bindings[1].binding=1;
        resolve_bindings[1].descriptorType=vk::DescriptorType::eStorageImage;
        for(auto& binding:resolve_bindings) {
            binding.descriptorCount=1;
            binding.stageFlags=vk::ShaderStageFlagBits::eCompute;
        }
        resolve_sets_.create(std::move(resolve_bindings),
                             static_cast<std::uint32_t>(depth.size()));
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
            resolve_sets_.write_buffer(i,0,outputs_[i]->descriptor());
            if(!hdr[i])throw std::invalid_argument("optics: missing HDR caustic view");
            vk::DescriptorImageInfo target{};
            target.imageView=hdr[i];
            target.imageLayout=vk::ImageLayout::eGeneral;
            resolve_sets_.write_image(i,1,target);
        }
        program_.create(shader,{sets_.layout()},sizeof(vk::util::math::mat4)+sizeof(std::array<std::uint32_t,4>));
        resolve_.create(resolve,{resolve_sets_.layout()},0);
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
    std::fill(valid_views_.begin(),valid_views_.end(),0u);
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
    // Photon deposits are in world coordinates. A still camera can reuse
    // its previous projected footprint instead of clearing/reprojecting
    // the same photons each frame. The HDR resolve still runs because
    // PCloud recomputes the lighting image every frame.
    if(valid_views_[image] &&
       std::memcmp(&cached_views_[image],&vp,sizeof(vp))==0) {
        if(source_.capacity) {
            vk::BufferMemoryBarrier visible{};
            visible.srcAccessMask=vk::AccessFlagBits::eShaderWrite;
            visible.dstAccessMask=vk::AccessFlagBits::eShaderRead;
            visible.srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
            visible.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
            visible.buffer=out.handle();
            visible.size=VK_WHOLE_SIZE;
            cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                vk::PipelineStageFlagBits::eComputeShader,{},
                                {},{visible},{});
            resolve_.dispatch(cmd,(extent_.width+7u)/8u,
                              (extent_.height+7u)/8u,1u,
                              {resolve_sets_.set(image)});
        }
        return;
    }
    cached_views_[image]=vp;
    valid_views_[image]=1u;
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
                        vk::PipelineStageFlagBits::eComputeShader,{},
                        {},{after},{});
    if(source_.capacity)
        resolve_.dispatch(cmd,(extent_.width+7u)/8u,
                          (extent_.height+7u)/8u,1u,
                          {resolve_sets_.set(image)});
}
vk::Buffer caustics::buffer(std::uint32_t image) const
{
    return outputs_.at(image)->handle();
}
void caustics::reset() noexcept
{
    resolve_.reset();
    program_.reset();
    resolve_sets_.reset();
    sets_.reset();
    outputs_.clear();
    cached_views_.clear();
    valid_views_.clear();
    if(sampler_)vk::device::get()->handle.destroySampler(sampler_);
    sampler_=nullptr;
    source_={};
    extent_=vk::Extent2D{};
}
}
