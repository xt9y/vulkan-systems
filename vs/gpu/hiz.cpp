#include <vs/gpu/hiz.hpp>
#include <algorithm>
#include <stdexcept>

namespace vs::gpu {
void depth_pyramid::create(vk::ImageView source,vk::Extent2D size,
                           vk::ImageLayout layout,
                           const std::string& shader,bool reverse_z)
{
    if(levels_ || !source || !size.width || !size.height)
        throw std::invalid_argument("Invalid depth pyramid configuration");
    const auto device=vk::device::get()->handle;
    extent_=vk::Extent2D{std::max(1u,(size.width+1u)/2u),
                         std::max(1u,(size.height+1u)/2u)};
    reverse_=reverse_z?1u:0u;
    auto width=extent_.width,height=extent_.height;
    levels_=1u;
    while(width>1u || height>1u) {
        width=std::max(1u,width/2u);
        height=std::max(1u,height/2u);
        ++levels_;
    }
    try {
        const auto features=vk::physical_device::get()->handle
            .getFormatProperties(vk::Format::eR32Sfloat).optimalTilingFeatures;
        const auto required=vk::FormatFeatureFlagBits::eStorageImage |
                            vk::FormatFeatureFlagBits::eSampledImage;
        if((features&required)!=required)
            throw std::runtime_error("R32F depth pyramid unsupported");
        vk::ImageCreateInfo info{};
        info.imageType=vk::ImageType::e2D;
        info.extent=vk::Extent3D{extent_.width,extent_.height,1u};
        info.mipLevels=levels_;
        info.arrayLayers=1;
        info.format=vk::Format::eR32Sfloat;
        info.tiling=vk::ImageTiling::eOptimal;
        info.samples=vk::SampleCountFlagBits::e1;
        info.usage=vk::ImageUsageFlagBits::eStorage |
                   vk::ImageUsageFlagBits::eSampled;
        image_.create(info,vk::MemoryPropertyFlagBits::eDeviceLocal);
        vk::ImageViewCreateInfo all{};
        all.viewType=vk::ImageViewType::e2D;
        all.format=info.format;
        all.subresourceRange={vk::ImageAspectFlagBits::eColor,0,levels_,0,1};
        image_.make_view(all);
        for(std::uint32_t i=0;i<levels_;++i) {
            vk::ImageViewCreateInfo mip{};
            mip.image=image_.handle();
            mip.viewType=vk::ImageViewType::e2D;
            mip.format=info.format;
            mip.subresourceRange={vk::ImageAspectFlagBits::eColor,i,1u,0,1};
            level_views_.push_back(device.createImageView(mip));
        }
        vk::SamplerCreateInfo sampler{};
        sampler.magFilter=vk::Filter::eNearest;
        sampler.minFilter=vk::Filter::eNearest;
        sampler.mipmapMode=vk::SamplerMipmapMode::eNearest;
        sampler.addressModeU=vk::SamplerAddressMode::eClampToEdge;
        sampler.addressModeV=vk::SamplerAddressMode::eClampToEdge;
        sampler.addressModeW=vk::SamplerAddressMode::eClampToEdge;
        sampler.maxLod=float(levels_);
        sampler_=device.createSampler(sampler);
        std::vector<vk::DescriptorSetLayoutBinding> bindings(2);
        bindings[0].binding=0;
        bindings[0].descriptorType=vk::DescriptorType::eCombinedImageSampler;
        bindings[0].descriptorCount=1;
        bindings[0].stageFlags=vk::ShaderStageFlagBits::eCompute;
        bindings[1].binding=1;
        bindings[1].descriptorType=vk::DescriptorType::eStorageImage;
        bindings[1].descriptorCount=1;
        bindings[1].stageFlags=vk::ShaderStageFlagBits::eCompute;
        descriptors_.create(std::move(bindings),levels_);
        for(std::uint32_t i=0;i<levels_;++i) {
            vk::DescriptorImageInfo input{};
            input.sampler=sampler_;
            input.imageView=i?level_views_[i-1]:source;
            input.imageLayout=i?vk::ImageLayout::eGeneral:layout;
            descriptors_.write_image(i,0,input);
            vk::DescriptorImageInfo output{};
            output.imageView=level_views_[i];
            output.imageLayout=vk::ImageLayout::eGeneral;
            descriptors_.write_image(i,1,output);
        }
        kernel_.create(shader,{descriptors_.layout()},sizeof(std::uint32_t));
    }catch(...) {reset();throw;}
}
void depth_pyramid::build(vk::CommandBuffer command)
{
    if(!levels_ || !kernel_.handle() || !command)
        throw std::logic_error("Depth pyramid not created");
    if(!initialized_) {
        vk::ImageMemoryBarrier barrier{};
        barrier.image=image_.handle();
        barrier.oldLayout=vk::ImageLayout::eUndefined;
        barrier.newLayout=vk::ImageLayout::eGeneral;
        barrier.dstAccessMask=vk::AccessFlagBits::eShaderWrite;
        barrier.subresourceRange={vk::ImageAspectFlagBits::eColor,0,levels_,0,1};
        command.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
                                vk::PipelineStageFlagBits::eComputeShader,
                                {},{},{},{barrier});
        initialized_=true;
    }
    // The same frame may have sampled the previous pyramid before this
    // update. Serialize those compute reads against the upcoming writes.
    vk::ImageMemoryBarrier reuse{};
    reuse.image=image_.handle();
    reuse.oldLayout=vk::ImageLayout::eGeneral;
    reuse.newLayout=vk::ImageLayout::eGeneral;
    reuse.srcAccessMask=vk::AccessFlagBits::eShaderRead;
    reuse.dstAccessMask=vk::AccessFlagBits::eShaderWrite;
    reuse.subresourceRange={vk::ImageAspectFlagBits::eColor,0,levels_,0,1};
    command.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                            vk::PipelineStageFlagBits::eComputeShader,
                            {},{},{},{reuse});
    auto width=extent_.width,height=extent_.height;
    for(std::uint32_t i=0;i<levels_;++i) {
        kernel_.dispatch(command,(width+7u)/8u,(height+7u)/8u,1u,
                         {descriptors_.set(i)},&reverse_,sizeof(reverse_));
        vk::ImageMemoryBarrier barrier{};
        barrier.image=image_.handle();
        barrier.oldLayout=vk::ImageLayout::eGeneral;
        barrier.newLayout=vk::ImageLayout::eGeneral;
        barrier.srcAccessMask=vk::AccessFlagBits::eShaderWrite;
        barrier.dstAccessMask=vk::AccessFlagBits::eShaderRead |
                              vk::AccessFlagBits::eShaderWrite;
        barrier.subresourceRange={vk::ImageAspectFlagBits::eColor,i,1u,0,1};
        command.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                vk::PipelineStageFlagBits::eComputeShader,
                                {},{},{},{barrier});
        width=std::max(1u,width/2u);
        height=std::max(1u,height/2u);
    }
}
void depth_pyramid::reset() noexcept
{
    try {
        kernel_.reset();
        if(levels_)descriptors_.reset();
        const auto device=vk::device::get()->handle;
        for(auto view:level_views_)if(view)device.destroyImageView(view);
        level_views_.clear();
        if(sampler_)device.destroySampler(sampler_);
        sampler_=vk::Sampler{};
        image_.reset();
    }catch(...) {
        level_views_.clear();
        sampler_=vk::Sampler{};
    }
    levels_=0;reverse_=0;initialized_=false;extent_=vk::Extent2D{};
}
} // namespace vs::gpu
