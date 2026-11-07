#include "lumen.hpp"
#include "shaders.hpp"
#include "trace.hpp"

#include <gpu/buffer.hpp>
#include <gpu/image.hpp>
#include <gpu/descriptors.hpp>
#include <gpu/compute.hpp>
#include <cstdint>
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
#include <iostream>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace lumen {
namespace {

using point=std::array<float,3>;

struct vertex {
    point position,normal,albedo,emission;
    std::uint32_t face_index=0;
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
    for(std::size_t i=0;i<vertices.size();i++)
        vertices[i].face_index=std::uint32_t(i/6);
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

// Keep these fixed bounds consistent with lumen/shaders.hpp.
// Only a bounded fraction of the 12x8x12 radiance field is refreshed each frame.
constexpr std::uint32_t radiance_probe_count=12*8*12;
constexpr std::uint32_t radiance_update_count=96;
constexpr vk::DeviceSize radiance_cache_bytes=
    vk::DeviceSize(radiance_probe_count)*4u*4u*sizeof(float);
struct radiance_constants {
    std::uint32_t frame=0,revision=0;
};
static_assert(sizeof(radiance_constants)==8);

// Shared BVH traversal implementation is inserted into both shader entrypoints.
// It is kept in one GLSL definition, not duplicated across compute/fragment.
std::string with_tracing(const char* source)
{
    std::string result(source);
    const std::string marker="/* LUMEN_SHARED_TRACING */";
    const auto offset=result.find(marker);
    if(offset==std::string::npos)
        throw std::runtime_error("missing shared Lumen ray traversal marker");
    result.replace(offset,marker.size(),shaders::shared_tracing);
    return result;
}

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
    vk::Extent2D render_extent{};
    scene geometry{};
    lighting lights{};
    gbuffer surfaces{};

    gpu::buffer vertex_buffer,faces_buffer,trace_nodes,trace_triangles;
    gpu::buffer radiance_probes;
    gpu::image color,depth,surface_cache;
    render::target target;
    render::pipeline geometry_pipeline,composite_pipeline;
    gpu::compute cache_pipeline,radiance_pipeline;
    gpu::descriptors geometry_sets,composite_sets,cache_sets,radiance_sets;
    vk::PipelineLayout geometry_layout{},composite_layout{};
    vk::Sampler sampler{};
    render::mesh box_mesh,fullscreen;
    render::material box_material,screen_material;
    std::uint32_t vertex_count=0;
    vk::QueryPool timestamp_pool{};
    std::uint32_t timestamp_bits=0;
    float timestamp_period=0.f;
    performance gpu_stats{};
    bool timestamp_pending=false;
    bool ready=false;
    bool cache_valid=false,cache_dirty=true;
    bool radiance_initialized=false;
    std::string geometry_vertex_spv,geometry_fragment_spv;
    std::string composite_vertex_spv,composite_fragment_spv,cache_spv,radiance_spv;

    ~implementation(){release();}

    void release() noexcept {
        ready=false;
        try {
            if(timestamp_pool)vk::query::destroy(timestamp_pool);
            timestamp_pool=vk::QueryPool{};
            gpu_stats={};
            timestamp_pending=false;
            box_material={};
            screen_material={};
            geometry_pipeline.reset();
            composite_pipeline.reset();
            cache_pipeline.reset();
            radiance_pipeline.reset();
            if(geometry_layout)vk::pipeline_layout::destroy(geometry_layout);
            if(composite_layout)vk::pipeline_layout::destroy(composite_layout);
            geometry_layout=vk::PipelineLayout{};
            composite_layout=vk::PipelineLayout{};
            composite_sets.reset();
            geometry_sets.reset();
            cache_sets.reset();
            radiance_sets.reset();
            if(sampler)vk::device::get()->handle.destroySampler(sampler);
            sampler=vk::Sampler{};
            target.reset(); // framebuffer must be destroyed before image views
            depth.reset();
            color.reset();
            surface_cache.reset();
            radiance_probes.reset();
            radiance_initialized=false;
            cache_valid=false;
            cache_dirty=true;
            vertex_buffer.reset();
            faces_buffer.reset();
            trace_nodes.reset();
            trace_triangles.reset();
            box_mesh={};
            fullscreen={};
            vertex_count=0;
            render_extent=vk::Extent2D{};
        } catch(...) {}
    }

    void initialize() {
        if(options.demo!=demo_scene::cornell_box)
            throw std::logic_error("Lumen Stage 1 supports only the Cornell demo");
        if(!presentation_pass || !extent.width || !extent.height)
            throw std::invalid_argument("invalid Lumen host presentation target");

        // The GLFW window may use a 2x Retina backing store. Never
        // brute-force per-fragment ray queries at that native resolution.
        // Composite still covers the full host swapchain extent.
        const float scale=std::clamp(options.tracing_scale,.25f,1.f);
        const float factor=std::min(scale,1280.f/float(extent.width));
        render_extent=vk::Extent2D{
            std::max(1u,std::uint32_t(std::lround(extent.width*factor))),
            std::max(1u,std::uint32_t(std::lround(extent.height*factor)))
        };
        const auto physical=vk::physical_device::get()->handle;
        const auto color_format=vk::Format::eR16G16B16A16Sfloat;
        const auto card_format=vk::Format::eR8G8B8A8Unorm;
        const auto depth_format=vk::Format::eD32Sfloat;
        const auto color_features=physical.getFormatProperties(color_format)
                                      .optimalTilingFeatures;
        const auto depth_features=physical.getFormatProperties(depth_format)
                                      .optimalTilingFeatures;
        const auto card_features=physical.getFormatProperties(card_format)
                                      .optimalTilingFeatures;
        const auto card_required=vk::FormatFeatureFlagBits::eStorageImage |
                                 vk::FormatFeatureFlagBits::eSampledImage;
        if((card_features & card_required)!=card_required)
            throw std::runtime_error("RGBA8 material card storage/sampling unsupported");
        const auto requested_color=vk::FormatFeatureFlagBits::eColorAttachment |
                                   vk::FormatFeatureFlagBits::eSampledImage;
        if((color_features & requested_color)!=requested_color)
            throw std::runtime_error("RGBA16F color attachment sampling unsupported");
        if((depth_features & vk::FormatFeatureFlagBits::eDepthStencilAttachment)!=
           vk::FormatFeatureFlagBits::eDepthStencilAttachment)
            throw std::runtime_error("D32 depth attachment unsupported");

        const auto vertices=cornell_box();
        const auto faces=make_faces(vertices);
        std::vector<point> shadow_vertices;
        shadow_vertices.reserve(vertices.size()-6);
        // Exclude the ceiling emitter from shadow occluders.
        for(std::size_t i=0;i<vertices.size()-6;i++)
            shadow_vertices.push_back(vertices[i].position);
        const auto accel=trace::build(shadow_vertices);
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
        const auto storage=vk::BufferUsageFlagBits::eStorageBuffer;
        const auto host_memory=vk::MemoryPropertyFlagBits::eHostVisible |
                               vk::MemoryPropertyFlagBits::eHostCoherent;
        trace_nodes.create(accel.nodes.size()*sizeof(bvh::node),
                           storage,host_memory);
        trace_nodes.upload(accel.nodes.data(),
                           accel.nodes.size()*sizeof(bvh::node));
        trace_triangles.create(accel.triangles.size()*sizeof(bvh::triangle),
                               storage,host_memory);
        trace_triangles.upload(accel.triangles.data(),
                               accel.triangles.size()*sizeof(bvh::triangle));
        radiance_probes.create(radiance_cache_bytes,
            vk::BufferUsageFlagBits::eStorageBuffer |
            vk::BufferUsageFlagBits::eTransferDst,
            vk::MemoryPropertyFlagBits::eDeviceLocal);
        box_mesh.vertex_count=vertex_count;
        box_mesh.streams={{0,vertex_buffer.handle(),0}};
        fullscreen.vertex_count=3;

        make_image(color,render_extent,color_format,
            vk::ImageUsageFlagBits::eColorAttachment |
            vk::ImageUsageFlagBits::eSampled,
            vk::ImageAspectFlagBits::eColor);
        make_image(depth,render_extent,depth_format,
            vk::ImageUsageFlagBits::eDepthStencilAttachment,
            vk::ImageAspectFlagBits::eDepth);
        make_image(surface_cache,{256,256},card_format,
            vk::ImageUsageFlagBits::eStorage |
            vk::ImageUsageFlagBits::eSampled,
            vk::ImageAspectFlagBits::eColor);

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
        desc.extent=render_extent;
        target.create(desc);

        vk::DescriptorSetLayoutBinding scene_binding{};
        scene_binding.binding=0;
        scene_binding.descriptorType=vk::DescriptorType::eStorageBuffer;
        scene_binding.descriptorCount=1;
        scene_binding.stageFlags=vk::ShaderStageFlagBits::eFragment;
        auto nodes_binding=scene_binding;
        nodes_binding.binding=1;
        auto triangles_binding=scene_binding;
        triangles_binding.binding=2;
        auto cards_binding=scene_binding;
        cards_binding.binding=3;
        cards_binding.descriptorType=vk::DescriptorType::eCombinedImageSampler;
        auto probes_binding=scene_binding;
        probes_binding.binding=4;
        geometry_sets.create({scene_binding,nodes_binding,triangles_binding,
                              cards_binding,probes_binding});
        geometry_sets.write_buffer(0,0,faces_buffer.descriptor());
        geometry_sets.write_buffer(0,1,trace_nodes.descriptor());
        geometry_sets.write_buffer(0,2,trace_triangles.descriptor());
        geometry_sets.write_buffer(0,4,radiance_probes.descriptor());

        auto cache_storage=scene_binding;
        cache_storage.binding=1;
        cache_storage.descriptorType=vk::DescriptorType::eStorageImage;
        cache_storage.stageFlags=vk::ShaderStageFlagBits::eCompute;
        auto cache_faces=scene_binding;
        cache_faces.stageFlags=vk::ShaderStageFlagBits::eCompute;
        cache_sets.create({cache_faces,cache_storage});
        cache_sets.write_buffer(0,0,faces_buffer.descriptor());
        vk::DescriptorImageInfo cache_write{};
        cache_write.imageView=surface_cache.view();
        cache_write.imageLayout=vk::ImageLayout::eGeneral;
        cache_sets.write_image(0,1,cache_write);

        auto radiance_faces=scene_binding;
        auto radiance_nodes=nodes_binding;
        auto radiance_triangles=triangles_binding;
        auto radiance_cards=cards_binding;
        auto radiance_buffer=probes_binding;
        radiance_faces.stageFlags=vk::ShaderStageFlagBits::eCompute;
        radiance_nodes.stageFlags=vk::ShaderStageFlagBits::eCompute;
        radiance_triangles.stageFlags=vk::ShaderStageFlagBits::eCompute;
        radiance_cards.stageFlags=vk::ShaderStageFlagBits::eCompute;
        radiance_buffer.stageFlags=vk::ShaderStageFlagBits::eCompute;
        radiance_sets.create({radiance_faces,radiance_nodes,radiance_triangles,
                              radiance_cards,radiance_buffer});
        radiance_sets.write_buffer(0,0,faces_buffer.descriptor());
        radiance_sets.write_buffer(0,1,trace_nodes.descriptor());
        radiance_sets.write_buffer(0,2,trace_triangles.descriptor());
        radiance_sets.write_buffer(0,4,radiance_probes.descriptor());

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
        vk::DescriptorImageInfo cards_read{};
        cards_read.sampler=sampler;
        cards_read.imageView=surface_cache.view();
        cards_read.imageLayout=vk::ImageLayout::eShaderReadOnlyOptimal;
        geometry_sets.write_image(0,3,cards_read);
        radiance_sets.write_image(0,3,cards_read);

        cache_spv=compile("cards","comp",shaders::cards_comp);
        cache_pipeline.create(cache_spv,{cache_sets.layout()});
        const std::string radiance_source=with_tracing(shaders::radiance_comp);
        radiance_spv=compile("radiance","comp",radiance_source.c_str());
        radiance_pipeline.create(radiance_spv,{radiance_sets.layout()},
                                sizeof(radiance_constants));
        geometry_vertex_spv=compile("geometry","vert",shaders::geometry_vert);
        const std::string fragment_source=with_tracing(shaders::geometry_frag);
        geometry_fragment_spv=compile("geometry","frag",fragment_source.c_str());
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
                std::uint32_t(offsetof(vertex,emission))},
            vk::VertexInputAttributeDescription{
                4,0,vk::Format::eR32Uint,
                std::uint32_t(offsetof(vertex,face_index))}
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
        // Optional GPU timestamps; independent of actual GI rendering.
        const auto family=vk::physical_device::find_queue(
            vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eCompute);
        const auto families=physical.getQueueFamilyProperties();
        if(family<families.size() && families[family].timestampValidBits>0)
        {
            timestamp_bits=families[family].timestampValidBits;
            timestamp_period=physical.getProperties().limits.timestampPeriod;
            vk::QueryPoolCreateInfo info{};
            info.queryType=vk::QueryType::eTimestamp;
            info.queryCount=4;
            timestamp_pool=vk::query::create(info);
            gpu_stats.supported=true;
        }
        ready=true;
        std::cout<<"[lumen] Stage 5: "<<accel.triangles.size()
                 <<" triangles / "<<accel.nodes.size()
                 <<" BVH nodes | cards=16 x 64x64"
                 <<" | world probes="<<radiance_probe_count
                 <<" updates/frame="<<radiance_update_count
                 <<" | scene render="<<render_extent.width<<"x"<<render_extent.height
                 <<" (swapchain "<<extent.width<<"x"<<extent.height<<")"
                 <<" | GPU timestamps "
                 <<(gpu_stats.supported?"supported":"unavailable")<<'\n';
    }

    float tick_ms(std::uint64_t start,std::uint64_t stop) const
    {
        const std::uint64_t mask=timestamp_bits>=64?~std::uint64_t(0):
            ((std::uint64_t(1)<<timestamp_bits)-1);
        return float((stop-start)&mask)*timestamp_period*1.0e-6f;
    }

    void begin_frame(vk::CommandBuffer cmd)
    {
        if(!timestamp_pool)return;
        if(timestamp_pending)
        {
            // Host has a single frame in flight; its fence has completed.
            std::array<std::uint64_t,4> ticks{};
            const auto status=vk::device::get()->handle.getQueryPoolResults(
                timestamp_pool,0,4,sizeof(ticks),ticks.data(),
                sizeof(std::uint64_t),vk::QueryResultFlagBits::e64);
            if(status==vk::Result::eSuccess)
            {
                gpu_stats.raster_ms=tick_ms(ticks[0],ticks[1]);
                gpu_stats.composite_ms=tick_ms(ticks[2],ticks[3]);
                gpu_stats.total_ms=tick_ms(ticks[0],ticks[3]);
                gpu_stats.valid=true;
            }
        }
        cmd.resetQueryPool(timestamp_pool,0,4);
        cmd.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe,
                           timestamp_pool,0);
        timestamp_pending=true;
    }
    void stamp(vk::CommandBuffer cmd,std::uint32_t index) const
    {
        if(timestamp_pool)
            cmd.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe,
                               timestamp_pool,index);
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
            float(render_extent.width)/float(render_extent.height),.05f,200.f
        }};

        vk::ClearValue black{},depth_clear{};
        black.color=vk::ClearColorValue(std::array<float,4>{0,0,0,1});
        depth_clear.depthStencil=vk::ClearDepthStencilValue{1.f,0};
        render::renderer recorder(cmd);
        recorder.begin_pass(target,vk::Rect2D{{0,0},render_extent},
                            {black,depth_clear});
        recorder.viewport(vk::Viewport{
            0,0,float(render_extent.width),float(render_extent.height),0.f,1.f},
            vk::Rect2D{{0,0},render_extent});
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

    // Stage 3+: reusable scene traversal and GI caches are not implemented yet.
    void update_surface_cache(vk::CommandBuffer cmd,const frame&)
    {
        if(!cache_dirty)return;
        vk::ImageMemoryBarrier barrier{};
        barrier.oldLayout=cache_valid?
            vk::ImageLayout::eShaderReadOnlyOptimal:vk::ImageLayout::eUndefined;
        barrier.newLayout=vk::ImageLayout::eGeneral;
        barrier.srcAccessMask=cache_valid?vk::AccessFlags(
            vk::AccessFlagBits::eShaderRead):vk::AccessFlags{};
        barrier.dstAccessMask=vk::AccessFlagBits::eShaderWrite;
        barrier.srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
        barrier.image=surface_cache.handle();
        barrier.subresourceRange={vk::ImageAspectFlagBits::eColor,0,1,0,1};
        cmd.pipelineBarrier(cache_valid?
                vk::PipelineStageFlags(vk::PipelineStageFlagBits::eFragmentShader):
                vk::PipelineStageFlags(vk::PipelineStageFlagBits::eTopOfPipe),
            vk::PipelineStageFlagBits::eComputeShader,{},{},{},{barrier});
        cache_pipeline.dispatch(cmd,32,32,1,{cache_sets.set()});
        barrier.oldLayout=vk::ImageLayout::eGeneral;
        barrier.newLayout=vk::ImageLayout::eShaderReadOnlyOptimal;
        barrier.srcAccessMask=vk::AccessFlagBits::eShaderWrite;
        barrier.dstAccessMask=vk::AccessFlagBits::eShaderRead;
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
            vk::PipelineStageFlagBits::eFragmentShader |
            vk::PipelineStageFlagBits::eComputeShader,{},{},{},{barrier});
        cache_valid=true;
        cache_dirty=false;
    }
    void trace_screen(vk::CommandBuffer,const frame&) {}
    void trace_scene(vk::CommandBuffer,const frame&) {}
    void update_screen_probes(vk::CommandBuffer,const frame&) {}
    void update_radiance_cache(vk::CommandBuffer cmd,const frame& current)
    {
        if(!radiance_initialized)
        {
            // Clear once: untouched probes are invalid (history count 0).
            // This avoids CPU roundtrips or uploading large zero-filled data.
            cmd.fillBuffer(radiance_probes.handle(),0,radiance_cache_bytes,0);
            vk::BufferMemoryBarrier init{};
            init.srcAccessMask=vk::AccessFlagBits::eTransferWrite;
            init.dstAccessMask=vk::AccessFlagBits::eShaderRead |
                               vk::AccessFlagBits::eShaderWrite;
            init.srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
            init.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
            init.buffer=radiance_probes.handle();
            init.offset=0;
            init.size=radiance_cache_bytes;
            cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                vk::PipelineStageFlagBits::eComputeShader,
                                {},{},{init},{});
            radiance_initialized=true;
        }
        const radiance_constants push{
            std::uint32_t(current.index),0u
        };
        radiance_pipeline.dispatch(cmd,
            (radiance_update_count+63u)/64u,1,1,
            {radiance_sets.set()},&push,sizeof(push));

        // Fragment shading consumes the newly updated coefficients. Every
        // probe owns a disjoint SSBO range, so the compute shader has no
        // cross-thread write conflicts or inter-probe dependencies.
        vk::BufferMemoryBarrier barrier{};
        barrier.srcAccessMask=vk::AccessFlagBits::eShaderWrite;
        barrier.dstAccessMask=vk::AccessFlagBits::eShaderRead;
        barrier.srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
        barrier.buffer=radiance_probes.handle();
        barrier.offset=0;
        barrier.size=radiance_cache_bytes;
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                            vk::PipelineStageFlagBits::eFragmentShader,
                            {},{},{barrier},{});
    }
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
    if(impl_)
    {
        impl_->cache_dirty=true;
        // Reinitialize all cached irradiance on the next record.
        impl_->radiance_initialized=false;
    }
}
void system::record(vk::CommandBuffer cmd,const frame& current) {
    if(!impl_ || !impl_->ready)return;
    impl_->begin_frame(cmd);
    impl_->update_surface_cache(cmd,current);
    impl_->update_radiance_cache(cmd,current);
    impl_->draw_scene(cmd,current);
    impl_->stamp(cmd,1);
    impl_->trace_screen(cmd,current);
    impl_->trace_scene(cmd,current);
    impl_->update_screen_probes(cmd,current);
    impl_->gather_indirect(cmd,current);
    impl_->trace_reflections(cmd,current);
}
void system::compose(vk::CommandBuffer cmd) {
    if(!impl_ || !impl_->ready)return;
    impl_->stamp(cmd,2);
    impl_->composite(cmd);
    impl_->stamp(cmd,3);
}
outputs system::result() const noexcept {return {};}
performance system::gpu_performance() const noexcept {
    return impl_?impl_->gpu_stats:performance{};
}
bool system::implemented() const noexcept {
    return impl_ && impl_->ready;
}
void system::reset() noexcept {impl_.reset();}

} // namespace lumen
