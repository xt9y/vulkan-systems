#include <vs/visual/inspector.hpp>
#include <algorithm>
#include <stdexcept>

namespace vs::visual
{
    namespace {
        struct push {
            std::array<float,16> vp{};
            std::array<float,4> grid{};
            std::array<std::uint32_t,4> meta{};
            std::array<float,4> viewport{};
        };
        static_assert(sizeof(push)==112);
        vk::PipelineColorBlendAttachmentState alpha_blend()
        {
            vk::PipelineColorBlendAttachmentState blend{};
            blend.blendEnable=true;
            blend.srcColorBlendFactor=vk::BlendFactor::eSrcAlpha;
            blend.dstColorBlendFactor=vk::BlendFactor::eOneMinusSrcAlpha;
            blend.colorBlendOp=vk::BlendOp::eAdd;
            blend.srcAlphaBlendFactor=vk::BlendFactor::eOne;
            blend.dstAlphaBlendFactor=vk::BlendFactor::eOneMinusSrcAlpha;
            blend.alphaBlendOp=vk::BlendOp::eAdd;
            blend.colorWriteMask=vk::ColorComponentFlagBits::eR|
                vk::ColorComponentFlagBits::eG|
                vk::ColorComponentFlagBits::eB|
                vk::ColorComponentFlagBits::eA;
            return blend;
        }
    }
    void inspector::create(vk::RenderPass pass,std::uint32_t frames)
    {
        if(!pass || !frames)throw std::invalid_argument("visual inspector: missing render pass");
        reset();
        frames_=frames;
        try {
            std::vector<vk::DescriptorSetLayoutBinding> bindings;
            for(std::uint32_t i=0;i<3;++i) {
                vk::DescriptorSetLayoutBinding b{};
                b.binding=i;
                b.descriptorCount=1;
                b.descriptorType=i==0?
                    vk::DescriptorType::eCombinedImageSampler:
                    vk::DescriptorType::eStorageBuffer;
                b.stageFlags=vk::ShaderStageFlagBits::eVertex;
                bindings.push_back(b);
            }
            sets_.create(std::move(bindings),frames);
            vk::PushConstantRange push_range{};
            push_range.stageFlags=vk::ShaderStageFlagBits::eVertex;
            push_range.size=sizeof(push);
            layout_.create({sets_.layout()},{push_range});
            vk::SamplerCreateInfo info{};
            info.magFilter=vk::Filter::eNearest;
            info.minFilter=vk::Filter::eNearest;
            info.mipmapMode=vk::SamplerMipmapMode::eNearest;
            info.addressModeU=info.addressModeV=info.addressModeW=
                vk::SamplerAddressMode::eClampToEdge;
            sampler_=vk::device::get()->handle.createSampler(info);
            ::vs::render::raster_state raster{};
            raster.pass=pass;
            raster.layout=layout_.handle();
            raster.cull=vk::CullModeFlagBits::eNone;
            raster.depth_test=false;
            raster.depth_write=false;
            raster.blend_attachments={alpha_blend()};
            const std::vector<::vs::render::shader_stage> stages{
                {vk::ShaderStageFlagBits::eVertex,
                 "build/shaders/visual_inspect.vert.spv"},
                {vk::ShaderStageFlagBits::eFragment,
                 "build/shaders/visual_inspect.frag.spv"}
            };
            raster.topology=vk::PrimitiveTopology::eLineList;
            line_.create(raster,stages);
            raster.topology=vk::PrimitiveTopology::eTriangleList;
            disc_.create(raster,stages);
        }catch(...) {reset();throw;}
    }
    void inspector::bind(const std::vector<source>& per_frame)
    {
        if(per_frame.size()!=frames_)
            throw std::invalid_argument("visual inspector: swapchain source count changed");
        for(std::uint32_t i=0;i<frames_;++i) {
            const auto& s=per_frame[i];
            if(!s.probes || !s.point_cloud || !s.bvh)
                throw std::invalid_argument("visual inspector: missing PCloud resources");
            vk::DescriptorImageInfo grid{};
            grid.sampler=sampler_;
            grid.imageView=s.probes;
            grid.imageLayout=vk::ImageLayout::eGeneral;
            sets_.write_image(i,0,grid);
            sets_.write_buffer(i,1,{s.point_cloud,0,VK_WHOLE_SIZE});
            sets_.write_buffer(i,2,{s.bvh,0,VK_WHOLE_SIZE});
        }
        sources_=per_frame;
    }
    void inspector::record(vk::CommandBuffer cmd,std::uint32_t image,
                           std::uint32_t mode,
                           const std::array<float,16>& projection,
                           vk::Extent2D extent) const
    {
        if(mode<1 || mode>3 || image>=sources_.size() || !cmd ||
           !extent.width || !extent.height)return;
        const auto& source=sources_[image];
        const auto count=mode==1?source.node_count:
            mode==2?source.extent.width*source.extent.height*source.extent.depth:
                    source.point_count;
        if(!count)return;
        push data{};
        data.vp=projection;
        data.grid=source.grid;
        data.meta={mode,source.extent.width,source.extent.height,source.extent.depth};
        data.viewport={float(extent.width),float(extent.height),
                       float(source.shared_nodes),0};
        cmd.bindPipeline(vk::PipelineBindPoint::eGraphics,
                         mode==1?line_.handle():disc_.handle());
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                               layout_.handle(),0,{sets_.set(image)},{});
        cmd.pushConstants(layout_.handle(),vk::ShaderStageFlagBits::eVertex,
                          0,sizeof(data),&data);
        cmd.setViewport(0,{vk::Viewport{0,0,float(extent.width),
                                          float(extent.height),0,1}});
        cmd.setScissor(0,{vk::Rect2D{{0,0},extent}});
        cmd.draw(mode==1?24:6,count,0,0);
    }
    void inspector::reset() noexcept
    {
        sources_.clear();
        frames_=0;
        line_.reset();disc_.reset();
        layout_.reset();
        sets_.reset();
        if(sampler_) {
            try{vk::device::get()->handle.destroySampler(sampler_);}catch(...){}
            sampler_=nullptr;
        }
    }
}
