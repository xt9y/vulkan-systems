#include "lumen.hpp"
#include "shaders.hpp"

#include <gpu/buffer.hpp>
#include <gpu/image.hpp>
#include <gpu/descriptors.hpp>
#include <render/target.hpp>
#include <render/pipeline.hpp>
#include <render/renderer.hpp>
#include <render/mesh.hpp>
#include <render/material.hpp>

#include <array>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace lumen {
namespace {

using point=std::array<float,3>;

struct vertex {
    point position;
    point color;
};

point add(point a,point b) {
    return {a[0]+b[0],a[1]+b[1],a[2]+b[2]};
}
point shade(point a,float factor) {
    return {a[0]*factor,a[1]*factor,a[2]*factor};
}
void quad(std::vector<vertex>& out,point p,point u,point v,point c) {
    const auto pu=add(p,u),pv=add(p,v),puv=add(pu,v);
    out.insert(out.end(),{{p,c},{pu,c},{puv,c},
                          {p,c},{puv,c},{pv,c}});
}
void box(std::vector<vertex>& out,point low,point high,point color) {
    const float x=high[0]-low[0],y=high[1]-low[1],z=high[2]-low[2];
    quad(out,low,{x,0,0},{0,y,0},shade(color,.85f));
    quad(out,{low[0],low[1],high[2]},{x,0,0},{0,y,0},
         shade(color,.65f));
    quad(out,{low[0],high[1],low[2]},{x,0,0},{0,0,z},
         shade(color,1.05f));
    quad(out,low,{0,0,z},{0,y,0},shade(color,.75f));
    quad(out,{high[0],low[1],low[2]},{0,0,z},{0,y,0},
         shade(color,.72f));
    quad(out,low,{x,0,0},{0,0,z},shade(color,.6f));
}
std::vector<vertex> cornell_box() {
    std::vector<vertex> vertices;
    vertices.reserve(144);
    const point white{.76f,.74f,.70f};
    quad(vertices,{0,0,5.59f},{5.56f,0,0},{0,5.48f,0},white);
    quad(vertices,{0,0,0},{5.56f,0,0},{0,0,5.59f},shade(white,.9f));
    quad(vertices,{0,5.48f,0},{5.56f,0,0},{0,0,5.59f},shade(white,.78f));
    quad(vertices,{0,0,0},{0,0,5.59f},{0,5.48f,0},{.65f,.07f,.055f});
    quad(vertices,{5.56f,0,0},{0,0,5.59f},{0,5.48f,0},{.055f,.50f,.09f});
    box(vertices,{1.13f,0,2.12f},{2.68f,3.05f,4.19f},white);
    box(vertices,{3.14f,0,1.47f},{4.47f,1.62f,3.06f},white);
    quad(vertices,{2.15f,5.465f,1.88f},{1.26f,0,0},{0,0,1.06f},
         {3.0f,3.0f,3.0f});
    return vertices;
}

// No shader files or shader compilation code are needed in vulkan-experiment.
// The runtime compiler only runs at create()/resize(), never per frame.
std::string compile(const char* name,const char* stage,const char* source) {
    namespace fs=std::filesystem;
    const fs::path dir=fs::path("build")/"shaders";
    fs::create_directories(dir);
    const fs::path input=dir/(std::string("lumen_")+name+"."+stage);
    const fs::path output=input.string()+".spv";
    {
        std::ofstream file(input,std::ios::binary|std::ios::trunc);
        if(!file || !(file<<source))
            throw std::runtime_error("cannot write Lumen shader: "+input.string());
    }
    const std::string command="glslangValidator -V -S "+std::string(stage)+
        " -o \""+output.string()+"\" \""+input.string()+"\"";
    if(std::system(command.c_str())!=0)
        throw std::runtime_error("Lumen shader compilation failed: "+input.string());
    return output.string();
}

struct alignas(16) camera_data {
    std::array<float,4> eye,right,up,projection;
};
static_assert(sizeof(camera_data)==64);

vk::ImageViewCreateInfo image_view(vk::Format format,vk::ImageAspectFlags aspect) {
    vk::ImageViewCreateInfo view{};
    view.viewType=vk::ImageViewType::e2D;
    view.format=format;
    view.subresourceRange={aspect,0,1,0,1};
    return view;
}
void make_image(gpu::image& image,vk::Extent2D size,vk::Format format,
                vk::ImageUsageFlags usage,vk::ImageAspectFlags aspect) {
    vk::ImageCreateInfo info{};
    info.imageType=vk::ImageType::e2D;
    info.format=format;
    info.extent=vk::Extent3D{size.width,size.height,1};
    info.mipLevels=1;
    info.arrayLayers=1;
    info.samples=vk::SampleCountFlagBits::e1;
    info.tiling=vk::ImageTiling::eOptimal;
    info.usage=usage;
    info.sharingMode=vk::SharingMode::eExclusive;
    image.create(info,vk::MemoryPropertyFlagBits::eDeviceLocal);
    image.make_view(image_view(format,aspect));
}
vk::PipelineLayout make_layout(vk::DescriptorSetLayout descriptor,
                                vk::ShaderStageFlags stage,
                                std::uint32_t push_bytes) {
    vk::PushConstantRange range{};
    range.stageFlags=stage;
    range.size=push_bytes;
    vk::PipelineLayoutCreateInfo info{};
    if(descriptor) {
        info.setLayoutCount=1;
        info.pSetLayouts=&descriptor;
    }
    if(push_bytes) {
        info.pushConstantRangeCount=1;
        info.pPushConstantRanges=&range;
    }
    return vk::pipeline_layout::create(info);
}

} // namespace

struct system::implementation {
    settings options{};
    vk::RenderPass presentation_pass{};
    vk::Format presentation_format=vk::Format::eUndefined;
    vk::Extent2D extent{};
    scene geometry{};
    lighting lights{};
    gbuffer surfaces{};

    gpu::buffer vertex_buffer;
    gpu::image color,depth;
    render::target target;
    render::pipeline geometry_pipeline,composite_pipeline;
    gpu::descriptors composite_sets;
    vk::PipelineLayout geometry_layout{},composite_layout{};
    vk::Sampler sampler{};
    render::mesh box_mesh,fullscreen;
    render::material box_material,screen_material;
    std::uint32_t vertex_count=0;
    bool ready=false;
    std::string geometry_vertex_spv,geometry_fragment_spv;
    std::string composite_vertex_spv,composite_fragment_spv;

    ~implementation(){release();}

    void release() noexcept {
        ready=false;
        try {
            box_material={};
            screen_material={};
            geometry_pipeline.reset();
            composite_pipeline.reset();
            if(geometry_layout)vk::pipeline_layout::destroy(geometry_layout);
            if(composite_layout)vk::pipeline_layout::destroy(composite_layout);
            geometry_layout=vk::PipelineLayout{};
            composite_layout=vk::PipelineLayout{};
            composite_sets.reset();
            if(sampler)vk::device::get()->handle.destroySampler(sampler);
            sampler=vk::Sampler{};
            target.reset(); // framebuffer must be destroyed before image views
            depth.reset();
            color.reset();
            vertex_buffer.reset();
            box_mesh={};
            fullscreen={};
            vertex_count=0;
        } catch(...) {}
    }

    void initialize() {
        if(options.demo!=demo_scene::cornell_box)
            throw std::logic_error("Lumen Stage 1 supports only the Cornell demo");
        if(!presentation_pass || !extent.width || !extent.height)
            throw std::invalid_argument("invalid Lumen host presentation target");

        const auto physical=vk::physical_device::get()->handle;
        const auto color_format=vk::Format::eR16G16B16A16Sfloat;
        const auto depth_format=vk::Format::eD32Sfloat;
        const auto color_features=physical.getFormatProperties(color_format)
                                      .optimalTilingFeatures;
        const auto depth_features=physical.getFormatProperties(depth_format)
                                      .optimalTilingFeatures;
        const auto requested_color=vk::FormatFeatureFlagBits::eColorAttachment |
                                   vk::FormatFeatureFlagBits::eSampledImage;
        if((color_features & requested_color)!=requested_color)
            throw std::runtime_error("RGBA16F color attachment sampling unsupported");
        if(!(depth_features & vk::FormatFeatureFlagBits::eDepthStencilAttachment))
            throw std::runtime_error("D32 depth attachment unsupported");

        const auto vertices=cornell_box();
        vertex_count=std::uint32_t(vertices.size());
        vertex_buffer.create(vertices.size()*sizeof(vertex),
            vk::BufferUsageFlagBits::eVertexBuffer,
            vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent);
        vertex_buffer.upload(vertices.data(),vertices.size()*sizeof(vertex));
        box_mesh.vertex_count=vertex_count;
        box_mesh.streams={{0,vertex_buffer.handle(),0}};
        fullscreen.vertex_count=3;

        make_image(color,extent,color_format,
            vk::ImageUsageFlagBits::eColorAttachment |
            vk::ImageUsageFlagBits::eSampled,
            vk::ImageAspectFlagBits::eColor);
        make_image(depth,extent,depth_format,
            vk::ImageUsageFlagBits::eDepthStencilAttachment,
            vk::ImageAspectFlagBits::eDepth);

        vk::AttachmentDescription color_attachment{};
        color_attachment.format=color_format;
        color_attachment.samples=vk::SampleCountFlagBits::e1;
        color_attachment.loadOp=vk::AttachmentLoadOp::eClear;
        color_attachment.storeOp=vk::AttachmentStoreOp::eStore;
        color_attachment.initialLayout=vk::ImageLayout::eUndefined;
        color_attachment.finalLayout=vk::ImageLayout::eShaderReadOnlyOptimal;
        vk::AttachmentDescription depth_attachment{};
        depth_attachment.format=depth_format;
        depth_attachment.samples=vk::SampleCountFlagBits::e1;
        depth_attachment.loadOp=vk::AttachmentLoadOp::eClear;
        depth_attachment.storeOp=vk::AttachmentStoreOp::eDontCare;
        depth_attachment.initialLayout=vk::ImageLayout::eUndefined;
        depth_attachment.finalLayout=vk::ImageLayout::eDepthStencilAttachmentOptimal;
        render::target_description desc{};
        desc.attachments={color_attachment,depth_attachment};
        desc.colors={{0,vk::ImageLayout::eColorAttachmentOptimal}};
        desc.depth=vk::AttachmentReference{
            1,vk::ImageLayout::eDepthStencilAttachmentOptimal};
        desc.views={color.view(),depth.view()};
        desc.extent=extent;
        target.create(desc);

        vk::DescriptorSetLayoutBinding image_binding{};
        image_binding.binding=0;
        image_binding.descriptorType=vk::DescriptorType::eCombinedImageSampler;
        image_binding.descriptorCount=1;
        image_binding.stageFlags=vk::ShaderStageFlagBits::eFragment;
        composite_sets.create({image_binding});
        vk::SamplerCreateInfo sampler_info{};
        sampler_info.magFilter=vk::Filter::eLinear;
        sampler_info.minFilter=vk::Filter::eLinear;
        sampler_info.mipmapMode=vk::SamplerMipmapMode::eNearest;
        sampler_info.addressModeU=vk::SamplerAddressMode::eClampToEdge;
        sampler_info.addressModeV=vk::SamplerAddressMode::eClampToEdge;
        sampler_info.addressModeW=vk::SamplerAddressMode::eClampToEdge;
        sampler=vk::device::get()->handle.createSampler(sampler_info);
        vk::DescriptorImageInfo image_info{};
        image_info.sampler=sampler;
        image_info.imageView=color.view();
        image_info.imageLayout=vk::ImageLayout::eShaderReadOnlyOptimal;
        composite_sets.write_image(0,0,image_info);

        geometry_vertex_spv=compile("geometry","vert",shaders::geometry_vert);
        geometry_fragment_spv=compile("geometry","frag",shaders::geometry_frag);
        composite_vertex_spv=compile("composite","vert",shaders::composite_vert);
        composite_fragment_spv=compile("composite","frag",shaders::composite_frag);

        geometry_layout=make_layout({},vk::ShaderStageFlagBits::eVertex,
                                    sizeof(camera_data));
        composite_layout=make_layout(composite_sets.layout(),
                                      vk::ShaderStageFlagBits::eFragment,
                                      sizeof(std::uint32_t));

        render::raster_state scene_state{};
        scene_state.pass=target.pass();
        scene_state.layout=geometry_layout;
        scene_state.cull=vk::CullModeFlagBits::eNone;
        scene_state.depth_test=true;
        scene_state.depth_write=true;
        scene_state.depth_compare=vk::CompareOp::eLess;
        scene_state.bindings={
            vk::VertexInputBindingDescription{0,std::uint32_t(sizeof(vertex)),
                                               vk::VertexInputRate::eVertex}
        };
        scene_state.attributes={
            vk::VertexInputAttributeDescription{
                0,0,vk::Format::eR32G32B32Sfloat,
                std::uint32_t(offsetof(vertex,position))},
            vk::VertexInputAttributeDescription{
                1,0,vk::Format::eR32G32B32Sfloat,
                std::uint32_t(offsetof(vertex,color))}
        };
        geometry_pipeline.create(scene_state,{
            {vk::ShaderStageFlagBits::eVertex,geometry_vertex_spv},
            {vk::ShaderStageFlagBits::eFragment,geometry_fragment_spv}
        });
        box_material.pipeline=geometry_pipeline.handle();
        box_material.layout=geometry_layout;
        box_material.push_stages=vk::ShaderStageFlagBits::eVertex;

        render::raster_state screen_state{};
        screen_state.pass=presentation_pass;
        screen_state.layout=composite_layout;
        screen_state.cull=vk::CullModeFlagBits::eNone;
        screen_state.depth_test=false;
        screen_state.depth_write=false;
        composite_pipeline.create(screen_state,{
            {vk::ShaderStageFlagBits::eVertex,composite_vertex_spv},
            {vk::ShaderStageFlagBits::eFragment,composite_fragment_spv}
        });
        screen_material.pipeline=composite_pipeline.handle();
        screen_material.layout=composite_layout;
        screen_material.sets={composite_sets.set()};
        screen_material.push_stages=vk::ShaderStageFlagBits::eFragment;
        ready=true;
    }

    void draw_scene(vk::CommandBuffer cmd,const frame& current) {
        const float yaw=current.view.yaw,pitch=current.view.pitch;
        const float cp=std::cos(pitch),sp=std::sin(pitch);
        const float sy=std::sin(yaw),cy=std::cos(yaw);
        const point forward{cp*sy,sp,-cp*cy};
        const point right{ -forward[2],0,forward[0] };
        const point up{ -sp*sy,cp,sp*cy };
        const auto& eye=current.view.position;
        camera_data camera{{
            eye.x,eye.y,eye.z,0
        },{
            right[0],right[1],right[2],0
        },{
            up[0],up[1],up[2],0
        },{
            std::tan(current.view.vertical_fov*3.14159265f/360.f),
            float(extent.width)/float(extent.height),.05f,200.f
        }};

        vk::ClearValue black{},depth_clear{};
        black.color=vk::ClearColorValue(std::array<float,4>{0,0,0,1});
        depth_clear.depthStencil=vk::ClearDepthStencilValue{1.f,0};
        render::renderer recorder(cmd);
        recorder.begin_pass(target,vk::Rect2D{{0,0},extent},
                            {black,depth_clear});
        recorder.viewport(vk::Viewport{
            0,0,float(extent.width),float(extent.height),0.f,1.f},
            vk::Rect2D{{0,0},extent});
        box_material.push_data=&camera;
        box_material.push_size=sizeof(camera);
        recorder.draw(box_mesh,box_material);
        box_material.push_data=nullptr;
        box_material.push_size=0;
        recorder.end_pass();

        // Render pass final layout already transitions color for sampling.
        // Explicit write->read dependency before the host presentation pass.
        vk::ImageMemoryBarrier barrier{};
        barrier.srcAccessMask=vk::AccessFlagBits::eColorAttachmentWrite;
        barrier.dstAccessMask=vk::AccessFlagBits::eShaderRead;
        barrier.oldLayout=vk::ImageLayout::eShaderReadOnlyOptimal;
        barrier.newLayout=vk::ImageLayout::eShaderReadOnlyOptimal;
        barrier.srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
        barrier.image=color.handle();
        barrier.subresourceRange={
            vk::ImageAspectFlagBits::eColor,0,1,0,1};
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eColorAttachmentOutput,
                            vk::PipelineStageFlagBits::eFragmentShader,
                            {},{},{},{barrier});
    }

    void composite(vk::CommandBuffer cmd) {
        // Called inside the window's compatible presentation render pass.
        const bool srgb=presentation_format==vk::Format::eB8G8R8A8Srgb ||
                        presentation_format==vk::Format::eR8G8B8A8Srgb;
        const std::uint32_t manual_gamma=srgb?0u:1u;
        screen_material.push_data=&manual_gamma;
        screen_material.push_size=sizeof(manual_gamma);
        render::renderer recorder(cmd);
        recorder.draw(fullscreen,screen_material);
        screen_material.push_data=nullptr;
        screen_material.push_size=0;
    }

    // Next stages, intentionally no-op for Stage 1.
    void update_surface_cache(vk::CommandBuffer,const frame&) {}
    void trace_screen(vk::CommandBuffer,const frame&) {}
    void trace_scene(vk::CommandBuffer,const frame&) {}
    void update_screen_probes(vk::CommandBuffer,const frame&) {}
    void update_radiance_cache(vk::CommandBuffer,const frame&) {}
    void gather_indirect(vk::CommandBuffer,const frame&) {}
    void trace_reflections(vk::CommandBuffer,const frame&) {}
};

system::system()=default;
system::~system()=default;
system::system(system&&) noexcept=default;
system& system::operator=(system&&) noexcept=default;

void system::create(vk::RenderPass pass,vk::Format format,
                    vk::Extent2D extent,const settings& options) {
    if(impl_)throw std::logic_error("VulkanLumen already created");
    auto next=std::make_unique<implementation>();
    next->options=options;
    next->presentation_pass=pass;
    next->presentation_format=format;
    next->extent=extent;
    next->initialize();
    impl_=std::move(next);
}
void system::resize(vk::RenderPass pass,vk::Format format,vk::Extent2D extent) {
    if(!impl_)throw std::logic_error("VulkanLumen not created");
    // Resize requires the caller to have waited for device idle.
    const auto options=impl_->options;
    auto next=std::make_unique<implementation>();
    next->options=options;
    next->presentation_pass=pass;
    next->presentation_format=format;
    next->extent=extent;
    next->geometry=impl_->geometry;
    next->lights=impl_->lights;
    next->surfaces=impl_->surfaces;
    next->initialize();
    impl_=std::move(next);
}
void system::set_scene(const scene& value) {
    if(impl_)impl_->geometry=value;
}
void system::set_lighting(const lighting& value) {
    if(impl_)impl_->lights=value;
}
void system::set_gbuffer(const gbuffer& value) {
    if(impl_)impl_->surfaces=value;
}
void system::invalidate() {
    // Stage 1 is stateless; future Lumen caches will be invalidated here.
}
void system::record(vk::CommandBuffer cmd,const frame& current) {
    if(!impl_ || !impl_->ready)return;
    impl_->draw_scene(cmd,current);
    impl_->update_surface_cache(cmd,current);
    impl_->trace_screen(cmd,current);
    impl_->trace_scene(cmd,current);
    impl_->update_screen_probes(cmd,current);
    impl_->update_radiance_cache(cmd,current);
    impl_->gather_indirect(cmd,current);
    impl_->trace_reflections(cmd,current);
}
void system::compose(vk::CommandBuffer cmd) {
    if(impl_ && impl_->ready)impl_->composite(cmd);
}
outputs system::result() const noexcept {return {};}
bool system::implemented() const noexcept {
    return impl_ && impl_->ready;
}
void system::reset() noexcept {impl_.reset();}

} // namespace lumen
