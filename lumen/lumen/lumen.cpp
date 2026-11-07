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
    point position,normal,albedo,emission;
};

struct alignas(16) face {
    std::array<float,4> origin,u,v,normal,albedo,emission,metric;
};
static_assert(sizeof(face)==112);

point add(point a,point b) {
    return {a[0]+b[0],a[1]+b[1],a[2]+b[2]};
}
point sub(point a,point b) {
    return {a[0]-b[0],a[1]-b[1],a[2]-b[2]};
}
point mul(point a,float t) {
    return {a[0]*t,a[1]*t,a[2]*t};
}
float dot(point a,point b) {
    return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
}
point cross(point a,point b) {
    return {a[1]*b[2]-a[2]*b[1],
            a[2]*b[0]-a[0]*b[2],
            a[0]*b[1]-a[1]*b[0]};
}
point normalized(point v) {
    const auto length=std::sqrt(dot(v,v));
    if(length<1e-8f)throw std::logic_error("degenerate Cornell normal");
    return mul(v,1.f/length);
}

// Identical room, Cornell blocks, material colors and emission to
// vulkan-experiment/path-tracer-bak. Quad winding is preserved verbatim.
void add_face(std::vector<vertex>& vertices,point origin,point u,point v,
              point normal,point albedo,point emission={}) {
    const std::array<point,6> points{
        origin,add(origin,u),add(add(origin,u),v),
        origin,add(add(origin,u),v),add(origin,v)
    };
    for(const auto& p:points)
        vertices.push_back({p,normal,albedo,emission});
}

void add_block(std::vector<vertex>& vertices,std::array<point,4> bottom,
               float height,point albedo) {
    const point vertical{0,height,0};
    const point e0=sub(bottom[1],bottom[0]);
    const point axis=normalized(e0);
    const point delta=sub(bottom[3],bottom[0]);
    const point e1=sub(delta,mul(axis,dot(delta,axis)));
    bottom[1]=add(bottom[0],e0);
    bottom[3]=add(bottom[0],e1);
    bottom[2]=add(bottom[1],e1);

    add_face(vertices,add(bottom[0],vertical),e0,e1,{0,1,0},albedo);
    for(int i=0;i<4;i++) {
        const point edge=sub(bottom[(i+1)%4],bottom[i]);
        add_face(vertices,bottom[i],edge,vertical,
                 normalized(cross(edge,vertical)),albedo);
    }
}

std::vector<vertex> cornell_box() {
    std::vector<vertex> vertices;
    vertices.reserve(16*6);
    const point white{.725f,.710f,.680f};
    const point red{.630f,.065f,.050f};
    const point green{.140f,.450f,.091f};
    const float width=5.55f,depth=5.592f,height=5.488f;

    add_face(vertices,{0,0,0},{width,0,0},{0,0,depth},{0,1,0},white);
    add_face(vertices,{0,height,0},{width,0,0},{0,0,depth},{0,-1,0},white);
    add_face(vertices,{width,0,0},{0,0,depth},{0,height,0},{-1,0,0},red);
    add_face(vertices,{0,0,0},{0,0,depth},{0,height,0},{1,0,0},green);
    add_face(vertices,{0,0,depth},{width,0,0},{0,height,0},{0,0,-1},white);
    add_block(vertices,{{{1.30f,0,.65f},{.82f,0,2.25f},
                         {2.40f,0,2.72f},{2.90f,0,1.14f}}},1.65f,white);
    add_block(vertices,{{{4.23f,0,2.47f},{2.65f,0,2.96f},
                         {3.14f,0,4.56f},{4.72f,0,4.06f}}},3.30f,white);
    add_face(vertices,{2.13f,5.484f,2.27f},{1.30f,0,0},{0,0,1.05f},
             {0,-1,0},{0,0,0},{17,16,14});
    if(vertices.size()!=16*6)
        throw std::logic_error("invalid Cornell geometry");
    return vertices;
}

std::array<face,16> make_faces(const std::vector<vertex>& vertices) {
    std::array<face,16> result{};
    for(std::size_t i=0;i<result.size();i++) {
        const auto& a=vertices[i*6];
        const point u=sub(vertices[i*6+1].position,a.position);
        const point v=sub(vertices[i*6+5].position,a.position);
        const float uu=dot(u,u),uv=dot(u,v),vv=dot(v,v);
        const float det=uu*vv-uv*uv;
        if(det<1e-8f)throw std::logic_error("degenerate Cornell face");
        const auto pack=[](point p)->std::array<float,4> {
            return {p[0],p[1],p[2],0};
        };
        result[i]={
            pack(a.position),pack(u),pack(v),pack(a.normal),
            pack(a.albedo),pack(a.emission),
            {vv/det,-uv/det,uu/det,std::sqrt(dot(cross(u,v),cross(u,v)))}
        };
    }
    return result;
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

    gpu::buffer vertex_buffer,faces_buffer;
    gpu::image color,depth;
    render::target target;
    render::pipeline geometry_pipeline,composite_pipeline;
    gpu::descriptors geometry_sets,composite_sets;
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
            geometry_sets.reset();
            if(sampler)vk::device::get()->handle.destroySampler(sampler);
            sampler=vk::Sampler{};
            target.reset(); // framebuffer must be destroyed before image views
            depth.reset();
            color.reset();
            vertex_buffer.reset();
            faces_buffer.reset();
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
        if((depth_features & vk::FormatFeatureFlagBits::eDepthStencilAttachment)!=
           vk::FormatFeatureFlagBits::eDepthStencilAttachment)
            throw std::runtime_error("D32 depth attachment unsupported");

        const auto vertices=cornell_box();
        const auto faces=make_faces(vertices);
        vertex_count=std::uint32_t(vertices.size());
        vertex_buffer.create(vertices.size()*sizeof(vertex),
            vk::BufferUsageFlagBits::eVertexBuffer,
            vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent);
        vertex_buffer.upload(vertices.data(),vertices.size()*sizeof(vertex));
        faces_buffer.create(sizeof(faces),vk::BufferUsageFlagBits::eStorageBuffer,
            vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent);
        faces_buffer.upload(faces.data(),sizeof(faces));
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

        vk::DescriptorSetLayoutBinding scene_binding{};
        scene_binding.binding=0;
        scene_binding.descriptorType=vk::DescriptorType::eStorageBuffer;
        scene_binding.descriptorCount=1;
        scene_binding.stageFlags=vk::ShaderStageFlagBits::eFragment;
        geometry_sets.create({scene_binding});
        geometry_sets.write_buffer(0,0,faces_buffer.descriptor());

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

        geometry_layout=make_layout(geometry_sets.layout(),
                                    vk::ShaderStageFlagBits::eVertex,
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
                std::uint32_t(offsetof(vertex,normal))},
            vk::VertexInputAttributeDescription{
                2,0,vk::Format::eR32G32B32Sfloat,
                std::uint32_t(offsetof(vertex,albedo))},
            vk::VertexInputAttributeDescription{
                3,0,vk::Format::eR32G32B32Sfloat,
                std::uint32_t(offsetof(vertex,emission))}
        };
        geometry_pipeline.create(scene_state,{
            {vk::ShaderStageFlagBits::eVertex,geometry_vertex_spv},
            {vk::ShaderStageFlagBits::eFragment,geometry_fragment_spv}
        });
        box_material.pipeline=geometry_pipeline.handle();
        box_material.layout=geometry_layout;
        box_material.sets={geometry_sets.set()};
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
        const point right{ cy,0,sy };
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
