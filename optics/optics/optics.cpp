#include "optics.hpp"
#include <stdexcept>
#include <vector>

namespace optics {
void compositor::create(vk::RenderPass pass,const std::vector<input>& frames,
                        const char* vertex,const char* fragment)
{
    if(!pass || frames.empty() || !vertex || !fragment)
        throw std::invalid_argument("optics: missing presentation pass, frames or shaders");
    reset();
    try {
        std::vector<vk::DescriptorSetLayoutBinding> bindings(6);
        for(std::uint32_t i=0;i<6;++i) {
            bindings[i].binding=i;
            bindings[i].descriptorCount=1;
            bindings[i].descriptorType=vk::DescriptorType::eCombinedImageSampler;
            bindings[i].stageFlags=vk::ShaderStageFlagBits::eFragment;
        }
        descriptors_.create(bindings,static_cast<std::uint32_t>(frames.size()));
        vk::SamplerCreateInfo sampler_info{};
        sampler_info.magFilter=vk::Filter::eNearest;
        sampler_info.minFilter=vk::Filter::eNearest;
        sampler_info.mipmapMode=vk::SamplerMipmapMode::eNearest;
        sampler_info.addressModeU=vk::SamplerAddressMode::eClampToEdge;
        sampler_info.addressModeV=vk::SamplerAddressMode::eClampToEdge;
        sampler_info.addressModeW=vk::SamplerAddressMode::eClampToEdge;
        sampler_=vk::device::get()->handle.createSampler(sampler_info);
        const vk::DescriptorSetLayout set_layout=descriptors_.layout();
        vk::PushConstantRange push{};
        push.stageFlags=vk::ShaderStageFlagBits::eFragment;
        push.size=sizeof(camera);
        vk::PipelineLayoutCreateInfo layout_info{};
        layout_info.setLayoutCount=1;
        layout_info.pSetLayouts=&set_layout;
        layout_info.pushConstantRangeCount=1;
        layout_info.pPushConstantRanges=&push;
        layout_=vk::device::get()->handle.createPipelineLayout(layout_info);
        render::raster_state raster{};
        raster.pass=pass;
        raster.layout=layout_;
        raster.cull=vk::CullModeFlagBits::eNone;
        raster.depth_test=false;
        raster.depth_write=false;
        vk::PipelineColorBlendAttachmentState blend{};
        blend.blendEnable=true;
        blend.srcColorBlendFactor=vk::BlendFactor::eSrcAlpha;
        blend.dstColorBlendFactor=vk::BlendFactor::eOneMinusSrcAlpha;
        blend.colorBlendOp=vk::BlendOp::eAdd;
        blend.srcAlphaBlendFactor=vk::BlendFactor::eOne;
        blend.dstAlphaBlendFactor=vk::BlendFactor::eZero;
        blend.alphaBlendOp=vk::BlendOp::eAdd;
        blend.colorWriteMask=vk::ColorComponentFlagBits::eR |
            vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB;
        raster.blend_attachments.push_back(blend);
        pipeline_.create(raster,{
            {vk::ShaderStageFlagBits::eVertex,vertex},
            {vk::ShaderStageFlagBits::eFragment,fragment}
        });
        for(std::uint32_t i=0;i<frames.size();++i) {
            const auto& f=frames[i];
            const std::array<vk::ImageView,6> views{
                f.lit,f.opaque_depth,f.normal,f.tint,f.material,f.material_depth
            };
            for(std::uint32_t binding=0;binding<views.size();++binding) {
                if(!views[binding])throw std::invalid_argument("optics: missing input view");
                vk::DescriptorImageInfo image{};
                image.sampler=sampler_;
                image.imageView=views[binding];
                image.imageLayout=binding==1 || binding==5 ?
                    vk::ImageLayout::eDepthStencilReadOnlyOptimal :
                    binding==0 ? vk::ImageLayout::eGeneral :
                    vk::ImageLayout::eShaderReadOnlyOptimal;
                descriptors_.write_image(i,binding,image);
            }
        }
        frame_count_=static_cast<std::uint32_t>(frames.size());
    }catch(...) {reset();throw;}
}
void compositor::record(vk::CommandBuffer cmd,std::uint32_t frame,const camera& view) const
{
    if(!cmd || !pipeline_.handle() || frame>=frame_count_)
        throw std::logic_error("optics: missing pipeline or invalid frame");
    cmd.bindPipeline(vk::PipelineBindPoint::eGraphics,pipeline_.handle());
    const vk::DescriptorSet set=descriptors_.set(frame);
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,layout_,0,{set},{});
    cmd.pushConstants(layout_,vk::ShaderStageFlagBits::eFragment,0,sizeof(camera),&view);
    cmd.draw(3,1,0,0);
}
void compositor::reset() noexcept
{
    pipeline_.reset();
    if(layout_)vk::device::get()->handle.destroyPipelineLayout(layout_);
    layout_=nullptr;
    descriptors_.reset();
    if(sampler_)vk::device::get()->handle.destroySampler(sampler_);
    sampler_=nullptr;
    frame_count_=0;
}
}
