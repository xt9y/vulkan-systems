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
        // Keep diffuse RGB untouched; albedo.w stores roughness for the
        // demo's low-cost, radiance-cache-based rough reflections.
        auto material=pack(a.albedo);
        material[3]=i==0?0.68f:(i>=5 && i<15?0.56f:0.92f);
        result[i]={
            pack(a.position),pack(u),pack(v),pack(a.normal),
            material,pack(a.emission),
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
struct light_constants {
    std::uint32_t card,enable_indirect,reserved0,reserved1;
};
static_assert(sizeof(light_constants)==16);

struct screen_constants {
    camera_data camera;
    std::array<float,4> trace_parameters;
};
static_assert(sizeof(screen_constants)==80);

// Keep these fixed bounds consistent with lumen/shaders.hpp.
// Only a bounded fraction of the 12x8x12 radiance field is refreshed each frame.
constexpr std::uint32_t radiance_probe_count=12*8*12;
constexpr vk::DeviceSize radiance_cache_bytes=
    vk::DeviceSize(radiance_probe_count)*4u*4u*sizeof(float);
struct radiance_constants {
    std::uint32_t frame=0,revision=0,updates_per_frame=0,rays_per_probe=0;
};
static_assert(sizeof(radiance_constants)==16);

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
    const std::string irradiance_marker="/* LUMEN_SHARED_IRRADIANCE */";
    const auto irradiance=result.find(irradiance_marker);
    if(irradiance!=std::string::npos)
        result.replace(irradiance,irradiance_marker.size(),
                       shaders::shared_irradiance);
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
    vk::Extent2D render_extent{},screen_extent{};
    camera_data current_camera{};
    scene geometry{};
    lighting lights{};
    gbuffer surfaces{};

    gpu::buffer vertex_buffer,faces_buffer,trace_nodes,trace_triangles;
    gpu::buffer radiance_probes;
    gpu::image color,depth,surface_cache,surface_lighting,direct_lighting;
    gpu::image screen_light,screen_meta,screen_gather,screen_filtered;
    gpu::image screen_specular,screen_specular_filtered;
    render::target target;
    render::pipeline geometry_pipeline,composite_pipeline;
    gpu::compute cache_pipeline,lighting_pipeline,radiance_pipeline,screen_pipeline;
    gpu::compute screen_filter_pipeline;
    gpu::descriptors geometry_sets,composite_sets,cache_sets,lighting_sets;
    gpu::descriptors radiance_sets,screen_sets,screen_filter_sets;
    vk::PipelineLayout geometry_layout{},composite_layout{};
    vk::Sampler sampler{},depth_sampler{};
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
    bool lighting_initialized=false;
    bool radiance_initialized=false;
    bool screen_initialized=false,filter_initialized=false;
    std::string geometry_vertex_spv,geometry_fragment_spv;
    std::string composite_vertex_spv,composite_fragment_spv,cache_spv;
    std::string lighting_spv,radiance_spv,screen_spv,filter_spv;

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
            lighting_pipeline.reset();
            radiance_pipeline.reset();
            screen_pipeline.reset();
            screen_filter_pipeline.reset();
            if(geometry_layout)vk::pipeline_layout::destroy(geometry_layout);
            if(composite_layout)vk::pipeline_layout::destroy(composite_layout);
            geometry_layout=vk::PipelineLayout{};
            composite_layout=vk::PipelineLayout{};
            composite_sets.reset();
            geometry_sets.reset();
            cache_sets.reset();
            lighting_sets.reset();
            radiance_sets.reset();
            screen_sets.reset();
            screen_filter_sets.reset();
            if(sampler)vk::device::get()->handle.destroySampler(sampler);
            if(depth_sampler)vk::device::get()->handle.destroySampler(depth_sampler);
            sampler=vk::Sampler{};
            depth_sampler=vk::Sampler{};
            target.reset(); // framebuffer must be destroyed before image views
            depth.reset();
            color.reset();
            surface_cache.reset();
            surface_lighting.reset();
            direct_lighting.reset();
            lighting_initialized=false;
            screen_light.reset();
            screen_meta.reset();
            screen_gather.reset();
            screen_filtered.reset();
            screen_specular.reset();
            screen_specular_filtered.reset();
            screen_initialized=false;
            filter_initialized=false;
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
            screen_extent=vk::Extent2D{};
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
        screen_extent=vk::Extent2D{
            (render_extent.width+7u)/8u,(render_extent.height+7u)/8u
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
                                   vk::FormatFeatureFlagBits::eSampledImage |
                                   vk::FormatFeatureFlagBits::eStorageImage;
        if((color_features & requested_color)!=requested_color)
            throw std::runtime_error("RGBA16F color/storage/sampling unsupported");
        const auto depth_required=vk::FormatFeatureFlagBits::eDepthStencilAttachment |
                                  vk::FormatFeatureFlagBits::eSampledImage;
        if((depth_features & depth_required)!=depth_required)
            throw std::runtime_error("sampled D32 depth attachment unsupported");

        const auto vertices=cornell_box();
        const auto faces=make_faces(vertices);
        std::vector<point> scene_vertices;
        scene_vertices.reserve(vertices.size());
        // Include the emissive panel in radiance rays. Shadow rays use a
        // shortened maximum distance, so the destination is not occluded.
        for(const auto& v:vertices)
            scene_vertices.push_back(v.position);
        const auto accel=trace::build(scene_vertices);
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
            vk::ImageUsageFlagBits::eDepthStencilAttachment |
            vk::ImageUsageFlagBits::eSampled,
            vk::ImageAspectFlagBits::eDepth);
        make_image(screen_light,
            vk::Extent2D{screen_extent.width*8u,screen_extent.height*8u},
            color_format,
            vk::ImageUsageFlagBits::eStorage |
            vk::ImageUsageFlagBits::eSampled,
            vk::ImageAspectFlagBits::eColor);
        make_image(screen_meta,screen_extent,color_format,
            vk::ImageUsageFlagBits::eStorage |
            vk::ImageUsageFlagBits::eSampled,
            vk::ImageAspectFlagBits::eColor);
        make_image(screen_gather,screen_extent,color_format,
            vk::ImageUsageFlagBits::eStorage |
            vk::ImageUsageFlagBits::eSampled,
            vk::ImageAspectFlagBits::eColor);
        make_image(screen_filtered,screen_extent,color_format,
            vk::ImageUsageFlagBits::eStorage |
            vk::ImageUsageFlagBits::eSampled,
            vk::ImageAspectFlagBits::eColor);
        make_image(screen_specular,screen_extent,color_format,
            vk::ImageUsageFlagBits::eStorage |
            vk::ImageUsageFlagBits::eSampled,
            vk::ImageAspectFlagBits::eColor);
        make_image(screen_specular_filtered,screen_extent,color_format,
            vk::ImageUsageFlagBits::eStorage |
            vk::ImageUsageFlagBits::eSampled,
            vk::ImageAspectFlagBits::eColor);
        make_image(surface_lighting,vk::Extent2D{512,512},color_format,
            vk::ImageUsageFlagBits::eStorage |
            vk::ImageUsageFlagBits::eSampled |
            vk::ImageUsageFlagBits::eTransferDst,
            vk::ImageAspectFlagBits::eColor);
        make_image(direct_lighting,vk::Extent2D{512,512},color_format,
            vk::ImageUsageFlagBits::eStorage |
            vk::ImageUsageFlagBits::eSampled |
            vk::ImageUsageFlagBits::eTransferDst,
            vk::ImageAspectFlagBits::eColor);
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
        depth_attachment.storeOp=vk::AttachmentStoreOp::eStore;
        depth_attachment.initialLayout=vk::ImageLayout::eUndefined;
        depth_attachment.finalLayout=vk::ImageLayout::eDepthStencilReadOnlyOptimal;
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
        auto direct_texture=cards_binding;
        direct_texture.binding=11;
        geometry_sets.create({scene_binding,nodes_binding,triangles_binding,
                              cards_binding,probes_binding,direct_texture});
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
        auto lighting_texture=radiance_cards;
        lighting_texture.binding=9;
        radiance_sets.create({radiance_faces,radiance_nodes,radiance_triangles,
                              radiance_cards,radiance_buffer,lighting_texture});

        auto lighting_output=radiance_cards;
        lighting_output.binding=10;
        lighting_output.descriptorType=vk::DescriptorType::eStorageImage;
        auto direct_output=lighting_output;
        direct_output.binding=11;
        lighting_sets.create({radiance_faces,radiance_nodes,radiance_triangles,
                              radiance_cards,radiance_buffer,lighting_output,
                              direct_output});
        lighting_sets.write_buffer(0,0,faces_buffer.descriptor());
        lighting_sets.write_buffer(0,1,trace_nodes.descriptor());
        lighting_sets.write_buffer(0,2,trace_triangles.descriptor());
        lighting_sets.write_buffer(0,4,radiance_probes.descriptor());
        radiance_sets.write_buffer(0,0,faces_buffer.descriptor());
        radiance_sets.write_buffer(0,1,trace_nodes.descriptor());
        radiance_sets.write_buffer(0,2,trace_triangles.descriptor());
        radiance_sets.write_buffer(0,4,radiance_probes.descriptor());

        vk::DescriptorSetLayoutBinding image_binding{};
        image_binding.binding=0;
        image_binding.descriptorType=vk::DescriptorType::eCombinedImageSampler;
        image_binding.descriptorCount=1;
        image_binding.stageFlags=vk::ShaderStageFlagBits::eFragment;
        std::vector<vk::DescriptorSetLayoutBinding> composite_bindings(5);
        for(std::uint32_t i=0;i<4;i++)
        {
            composite_bindings[i]=image_binding;
            composite_bindings[i].binding=i;
        }
        composite_sets.create(composite_bindings);

        std::vector<vk::DescriptorSetLayoutBinding> screen_bindings(12);
        for(std::uint32_t i=0;i<11;i++)
        {
            auto& binding=screen_bindings[i];
            binding.binding=i;
            binding.descriptorCount=1;
            binding.stageFlags=vk::ShaderStageFlagBits::eCompute;
            binding.descriptorType=(i==3 || i==5 || i==6 || i==9)?
                vk::DescriptorType::eCombinedImageSampler:
                ((i==7 || i==8 || i==10 || i==11)?
                    vk::DescriptorType::eStorageImage:
                    vk::DescriptorType::eStorageBuffer);
        }
        screen_sets.create(screen_bindings);
        std::array<vk::DescriptorSetLayoutBinding,5> filter_bindings{};
        for(std::uint32_t i=0;i<filter_bindings.size();i++)
        {
            auto& binding=filter_bindings[i];
            binding.binding=i;
            binding.descriptorCount=1;
            binding.stageFlags=vk::ShaderStageFlagBits::eCompute;
            binding.descriptorType=(i==2 || i==4)?
                vk::DescriptorType::eStorageImage:
                vk::DescriptorType::eCombinedImageSampler;
        }
        screen_filter_sets.create(
            std::vector<vk::DescriptorSetLayoutBinding>(
                filter_bindings.begin(),filter_bindings.end()));
        screen_sets.write_buffer(0,0,faces_buffer.descriptor());
        screen_sets.write_buffer(0,1,trace_nodes.descriptor());
        screen_sets.write_buffer(0,2,trace_triangles.descriptor());
        screen_sets.write_buffer(0,4,radiance_probes.descriptor());

        vk::SamplerCreateInfo sampler_info{};
        sampler_info.magFilter=vk::Filter::eLinear;
        sampler_info.minFilter=vk::Filter::eLinear;
        sampler_info.mipmapMode=vk::SamplerMipmapMode::eNearest;
        sampler_info.addressModeU=vk::SamplerAddressMode::eClampToEdge;
        sampler_info.addressModeV=vk::SamplerAddressMode::eClampToEdge;
        sampler_info.addressModeW=vk::SamplerAddressMode::eClampToEdge;
        sampler=vk::device::get()->handle.createSampler(sampler_info);
        sampler_info.minFilter=vk::Filter::eNearest;
        sampler_info.magFilter=vk::Filter::eNearest;
        depth_sampler=vk::device::get()->handle.createSampler(sampler_info);
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
        screen_sets.write_image(0,3,cards_read);
        lighting_sets.write_image(0,3,cards_read);

        vk::DescriptorImageInfo lit_read{};
        lit_read.sampler=sampler;
        lit_read.imageView=surface_lighting.view();
        lit_read.imageLayout=vk::ImageLayout::eShaderReadOnlyOptimal;
        radiance_sets.write_image(0,9,lit_read);
        screen_sets.write_image(0,9,lit_read);
        vk::DescriptorImageInfo lit_write{};
        lit_write.imageView=surface_lighting.view();
        lit_write.imageLayout=vk::ImageLayout::eGeneral;
        lighting_sets.write_image(0,10,lit_write);
        vk::DescriptorImageInfo direct_read{};
        direct_read.sampler=sampler;
        direct_read.imageView=direct_lighting.view();
        direct_read.imageLayout=vk::ImageLayout::eShaderReadOnlyOptimal;
        geometry_sets.write_image(0,11,direct_read);
        vk::DescriptorImageInfo direct_write{};
        direct_write.imageView=direct_lighting.view();
        direct_write.imageLayout=vk::ImageLayout::eGeneral;
        lighting_sets.write_image(0,11,direct_write);

        vk::DescriptorImageInfo depth_read{};
        depth_read.sampler=depth_sampler;
        depth_read.imageView=depth.view();
        depth_read.imageLayout=vk::ImageLayout::eDepthStencilReadOnlyOptimal;
        composite_sets.write_image(0,1,depth_read);
        screen_sets.write_image(0,6,depth_read);
        screen_sets.write_image(0,5,image_info);

        const auto screen_read=[&](const gpu::image& image,
                                   std::uint32_t binding)
        {
            vk::DescriptorImageInfo info{};
            info.sampler=sampler;
            info.imageView=image.view();
            info.imageLayout=vk::ImageLayout::eShaderReadOnlyOptimal;
            composite_sets.write_image(0,binding,info);
        };
        screen_read(screen_filtered,2);
        screen_read(screen_meta,3);
        screen_read(screen_specular_filtered,4);

        const auto filter_read=[&](const gpu::image& image,
                                   std::uint32_t binding)
        {
            vk::DescriptorImageInfo info{};
            info.sampler=depth_sampler;
            info.imageView=image.view();
            info.imageLayout=vk::ImageLayout::eShaderReadOnlyOptimal;
            screen_filter_sets.write_image(0,binding,info);
        };
        filter_read(screen_gather,0);
        filter_read(screen_meta,1);
        filter_read(screen_specular,3);
        vk::DescriptorImageInfo filtered_write{};
        filtered_write.imageView=screen_filtered.view();
        filtered_write.imageLayout=vk::ImageLayout::eGeneral;
        screen_filter_sets.write_image(0,2,filtered_write);
        vk::DescriptorImageInfo specular_filtered_write{};
        specular_filtered_write.imageView=screen_specular_filtered.view();
        specular_filtered_write.imageLayout=vk::ImageLayout::eGeneral;
        screen_filter_sets.write_image(0,4,specular_filtered_write);

        const auto screen_write=[&](const gpu::image& image,
                                    std::uint32_t binding)
        {
            vk::DescriptorImageInfo info{};
            info.imageView=image.view();
            info.imageLayout=vk::ImageLayout::eGeneral;
            screen_sets.write_image(0,binding,info);
        };
        screen_write(screen_light,7);
        screen_write(screen_meta,8);
        screen_write(screen_gather,10);
        screen_write(screen_specular,11);

        cache_spv=compile("cards","comp",shaders::cards_comp);
        cache_pipeline.create(cache_spv,{cache_sets.layout()});
        const std::string lighting_source=with_tracing(shaders::lighting_comp);
        lighting_spv=compile("surface_lighting","comp",lighting_source.c_str());
        lighting_pipeline.create(lighting_spv,{lighting_sets.layout()},
                                sizeof(light_constants));
        const std::string radiance_source=with_tracing(shaders::radiance_comp);
        radiance_spv=compile("radiance","comp",radiance_source.c_str());
        radiance_pipeline.create(radiance_spv,{radiance_sets.layout()},
                                sizeof(radiance_constants));
        const std::string screen_source=with_tracing(shaders::screen_comp);
        screen_spv=compile("screen","comp",screen_source.c_str());
        screen_pipeline.create(screen_spv,{screen_sets.layout()},
                               sizeof(screen_constants));
        filter_spv=compile("screen_filter","comp",shaders::screen_filter_comp);
        screen_filter_pipeline.create(filter_spv,{screen_filter_sets.layout()});
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
            info.queryCount=7;
            timestamp_pool=vk::query::create(info);
            gpu_stats.supported=true;
        }
        ready=true;
        std::cout<<"[lumen] Lit Surface Cache + directional Screen Radiance Cache: "
                 <<accel.triangles.size()
                 <<" triangles / "<<accel.nodes.size()
                 <<" BVH nodes | lit cards=16 x 128x128"
                 <<" | world probes="<<radiance_probe_count
                 <<" updates/frame="<<std::clamp(options.probe_updates_per_frame,
                                                    1u,radiance_probe_count)
                 <<" rays/update="<<std::clamp(options.rays_per_probe,1u,16u)
                 <<" | screen probes="<<screen_extent.width<<"x"<<screen_extent.height
                 <<" spacing=8px"
                 <<" directions=8x8"
                 <<" | rough specular=screen radiance reuse"
                 <<" | geometry-aware probe filter=3x3"
                 <<" trace=8 steps+"<<options.screen_trace_distance<<"m BVH fallback"
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
            std::array<std::uint64_t,7> ticks{};
            const auto status=vk::device::get()->handle.getQueryPoolResults(
                timestamp_pool,0,7,sizeof(ticks),ticks.data(),
                sizeof(std::uint64_t),vk::QueryResultFlagBits::e64);
            if(status==vk::Result::eSuccess)
            {
                gpu_stats.material_cache_ms=tick_ms(ticks[0],ticks[1]);
                gpu_stats.radiance_cache_ms=tick_ms(ticks[1],ticks[2]);
                gpu_stats.raster_ms=tick_ms(ticks[2],ticks[3]);
                gpu_stats.screen_probes_ms=tick_ms(ticks[3],ticks[4]);
                gpu_stats.composite_ms=tick_ms(ticks[5],ticks[6]);
                gpu_stats.total_ms=tick_ms(ticks[0],ticks[6]);
                gpu_stats.valid=true;
            }
        }
        cmd.resetQueryPool(timestamp_pool,0,7);
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
        current_camera=camera;

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

        // The render pass performs the final image layout transitions.
        // Make both color and depth attachment writes visible to the probe
        // compute shader and the final fullscreen composite.
        std::array<vk::ImageMemoryBarrier,2> barriers{};
        barriers[0].srcAccessMask=vk::AccessFlagBits::eColorAttachmentWrite;
        barriers[0].dstAccessMask=vk::AccessFlagBits::eShaderRead;
        barriers[0].oldLayout=vk::ImageLayout::eShaderReadOnlyOptimal;
        barriers[0].newLayout=vk::ImageLayout::eShaderReadOnlyOptimal;
        barriers[0].image=color.handle();
        barriers[0].subresourceRange={
            vk::ImageAspectFlagBits::eColor,0,1,0,1};
        barriers[1].srcAccessMask=vk::AccessFlagBits::eDepthStencilAttachmentWrite;
        barriers[1].dstAccessMask=vk::AccessFlagBits::eShaderRead;
        barriers[1].oldLayout=vk::ImageLayout::eDepthStencilReadOnlyOptimal;
        barriers[1].newLayout=vk::ImageLayout::eDepthStencilReadOnlyOptimal;
        barriers[1].image=depth.handle();
        barriers[1].subresourceRange={
            vk::ImageAspectFlagBits::eDepth,0,1,0,1};
        for(auto& barrier:barriers)
        {
            barrier.srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
        }
        cmd.pipelineBarrier(
            vk::PipelineStageFlagBits::eColorAttachmentOutput |
            vk::PipelineStageFlagBits::eEarlyFragmentTests |
            vk::PipelineStageFlagBits::eLateFragmentTests,
            vk::PipelineStageFlagBits::eComputeShader |
            vk::PipelineStageFlagBits::eFragmentShader,
            {},{},{},barriers);
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
                vk::PipelineStageFlags(vk::PipelineStageFlagBits::eFragmentShader |
                                       vk::PipelineStageFlagBits::eComputeShader):
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
    void update_surface_lighting(vk::CommandBuffer cmd,const frame& current)
    {
        std::array<vk::ImageMemoryBarrier,2> images{};
        const std::array<vk::Image,2> targets{
            surface_lighting.handle(),direct_lighting.handle()
        };
        for(std::size_t i=0;i<images.size();i++)
        {
            auto& barrier=images[i];
            barrier.image=targets[i];
            barrier.subresourceRange={
                vk::ImageAspectFlagBits::eColor,0,1,0,1
            };
            barrier.srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
            barrier.oldLayout=lighting_initialized?
                vk::ImageLayout::eShaderReadOnlyOptimal:
                vk::ImageLayout::eUndefined;
            barrier.newLayout=lighting_initialized?
                vk::ImageLayout::eGeneral:
                vk::ImageLayout::eTransferDstOptimal;
            barrier.srcAccessMask=lighting_initialized?
                vk::AccessFlags(vk::AccessFlagBits::eShaderRead):
                vk::AccessFlags{};
            barrier.dstAccessMask=lighting_initialized?
                vk::AccessFlags(vk::AccessFlagBits::eShaderWrite):
                vk::AccessFlags(vk::AccessFlagBits::eTransferWrite);
        }
        cmd.pipelineBarrier(lighting_initialized?
                vk::PipelineStageFlags(vk::PipelineStageFlagBits::eComputeShader |
                                       vk::PipelineStageFlagBits::eFragmentShader):
                vk::PipelineStageFlags(vk::PipelineStageFlagBits::eTopOfPipe),
            lighting_initialized?
                vk::PipelineStageFlags(vk::PipelineStageFlagBits::eComputeShader):
                vk::PipelineStageFlags(vk::PipelineStageFlagBits::eTransfer),
            {},{},{},images);
        if(!lighting_initialized)
        {
            const vk::ClearColorValue black(std::array<float,4>{0,0,0,0});
            for(const auto& barrier:images)
                cmd.clearColorImage(barrier.image,
                    vk::ImageLayout::eTransferDstOptimal,
                    black,{barrier.subresourceRange});
            for(auto& barrier:images)
            {
                barrier.oldLayout=vk::ImageLayout::eTransferDstOptimal;
                barrier.newLayout=vk::ImageLayout::eGeneral;
                barrier.srcAccessMask=vk::AccessFlagBits::eTransferWrite;
                barrier.dstAccessMask=vk::AccessFlagBits::eShaderWrite;
            }
            cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                vk::PipelineStageFlagBits::eComputeShader,
                {},{},{},images);
        }
        if(radiance_initialized)
        {
            vk::BufferMemoryBarrier probe_barrier{};
            probe_barrier.srcAccessMask=vk::AccessFlagBits::eShaderWrite;
            probe_barrier.dstAccessMask=vk::AccessFlagBits::eShaderRead;
            probe_barrier.srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
            probe_barrier.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
            probe_barrier.buffer=radiance_probes.handle();
            probe_barrier.offset=0;
            probe_barrier.size=radiance_cache_bytes;
            cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                vk::PipelineStageFlagBits::eComputeShader,
                {},{},{probe_barrier},{});
        }
        // One 128x128 lit card per frame amortizes four-sample soft shadowing and
        // indirect radiosity. Both atlases remain resident between frames.
        const light_constants push{
            std::uint32_t(current.index%16u),
            radiance_initialized?1u:0u,0u,0u
        };
        lighting_pipeline.dispatch(cmd,16,16,1,{lighting_sets.set()},
                                   &push,sizeof(push));
        for(auto& barrier:images)
        {
            barrier.oldLayout=vk::ImageLayout::eGeneral;
            barrier.newLayout=vk::ImageLayout::eShaderReadOnlyOptimal;
            barrier.srcAccessMask=vk::AccessFlagBits::eShaderWrite;
            barrier.dstAccessMask=vk::AccessFlagBits::eShaderRead;
        }
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
            vk::PipelineStageFlagBits::eComputeShader |
            vk::PipelineStageFlagBits::eFragmentShader,
            {},{},{},images);
        lighting_initialized=true;
    }

    void update_screen_probes(vk::CommandBuffer cmd,const frame& current)
    {
        std::array<vk::ImageMemoryBarrier,4> barriers{};
        const std::array<vk::Image,4> images{
            screen_light.handle(),screen_meta.handle(),screen_gather.handle(),
            screen_specular.handle()
        };
        for(std::size_t i=0;i<barriers.size();i++)
        {
            auto& barrier=barriers[i];
            barrier.srcAccessMask=screen_initialized?
                vk::AccessFlags(vk::AccessFlagBits::eShaderRead |
                                vk::AccessFlagBits::eShaderWrite):
                vk::AccessFlags{};
            barrier.dstAccessMask=vk::AccessFlagBits::eShaderRead |
                                  vk::AccessFlagBits::eShaderWrite;
            barrier.oldLayout=screen_initialized?
                vk::ImageLayout::eShaderReadOnlyOptimal:vk::ImageLayout::eUndefined;
            barrier.newLayout=vk::ImageLayout::eGeneral;
            barrier.srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
            barrier.image=images[i];
            barrier.subresourceRange={
                vk::ImageAspectFlagBits::eColor,0,1,0,1
            };
        }
        cmd.pipelineBarrier(screen_initialized?
                vk::PipelineStageFlags(vk::PipelineStageFlagBits::eFragmentShader |
                                       vk::PipelineStageFlagBits::eComputeShader):
                vk::PipelineStageFlags(vk::PipelineStageFlagBits::eTopOfPipe),
            vk::PipelineStageFlagBits::eComputeShader,
            {},{},{},barriers);
        const screen_constants push{
            current_camera,
            {std::clamp(options.screen_trace_distance,.5f,16.f),
             float(std::clamp(options.screen_rays_per_probe,1u,8u)),
             screen_initialized?1.f:0.f,float(current.index%4096u)}
        };
        screen_pipeline.dispatch(cmd,(screen_extent.width+7)/8,
                                 (screen_extent.height+7)/8,1,
                                 {screen_sets.set()},&push,sizeof(push));
        for(auto& barrier:barriers)
        {
            barrier.srcAccessMask=vk::AccessFlagBits::eShaderWrite;
            barrier.dstAccessMask=vk::AccessFlagBits::eShaderRead;
            barrier.oldLayout=vk::ImageLayout::eGeneral;
            barrier.newLayout=vk::ImageLayout::eShaderReadOnlyOptimal;
        }
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                            vk::PipelineStageFlagBits::eComputeShader |
                            vk::PipelineStageFlagBits::eFragmentShader,
                            {},{},{},barriers);
        screen_initialized=true;
    }
    void filter_screen_probes(vk::CommandBuffer cmd)
    {
        // The directional pass has already transitioned gather/meta into
        // shader-read layout. Filter on the smaller screen-probe grid before
        // the full-resolution composite, respecting face and world depth.
        std::array<vk::ImageMemoryBarrier,2> barriers{};
        const std::array<vk::Image,2> targets{
            screen_filtered.handle(),screen_specular_filtered.handle()
        };
        for(std::size_t i=0;i<barriers.size();i++)
        {
            auto& barrier=barriers[i];
            barrier.oldLayout=filter_initialized?
                vk::ImageLayout::eShaderReadOnlyOptimal:vk::ImageLayout::eUndefined;
            barrier.newLayout=vk::ImageLayout::eGeneral;
            barrier.srcAccessMask=filter_initialized?
                vk::AccessFlags(vk::AccessFlagBits::eShaderRead):vk::AccessFlags{};
            barrier.dstAccessMask=vk::AccessFlagBits::eShaderWrite;
            barrier.srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
            barrier.image=targets[i];
            barrier.subresourceRange={
                vk::ImageAspectFlagBits::eColor,0,1,0,1
            };
        }
        cmd.pipelineBarrier(filter_initialized?
                vk::PipelineStageFlags(vk::PipelineStageFlagBits::eFragmentShader):
                vk::PipelineStageFlags(vk::PipelineStageFlagBits::eTopOfPipe),
            vk::PipelineStageFlagBits::eComputeShader,
            {},{},{},barriers);
        screen_filter_pipeline.dispatch(cmd,(screen_extent.width+7u)/8u,
                                        (screen_extent.height+7u)/8u,1,
                                        {screen_filter_sets.set()});
        for(auto& barrier:barriers)
        {
            barrier.oldLayout=vk::ImageLayout::eGeneral;
            barrier.newLayout=vk::ImageLayout::eShaderReadOnlyOptimal;
            barrier.srcAccessMask=vk::AccessFlagBits::eShaderWrite;
            barrier.dstAccessMask=vk::AccessFlagBits::eShaderRead;
        }
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                            vk::PipelineStageFlagBits::eFragmentShader,
                            {},{},{},barriers);
        filter_initialized=true;
    }

    void update_radiance_cache(vk::CommandBuffer cmd,const frame& current)
    {
        if(radiance_initialized)
        {
            // The previous frame read these coefficients in the fragment
            // stage. Make previous compute writes visible before new probe
            // read/modify/write updates, including across queue submissions.
            vk::BufferMemoryBarrier reuse{};
            reuse.srcAccessMask=vk::AccessFlagBits::eShaderWrite;
            reuse.dstAccessMask=vk::AccessFlagBits::eShaderRead |
                                vk::AccessFlagBits::eShaderWrite;
            reuse.srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
            reuse.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
            reuse.buffer=radiance_probes.handle();
            reuse.offset=0;
            reuse.size=radiance_cache_bytes;
            cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                vk::PipelineStageFlagBits::eComputeShader,
                                {},{},{reuse},{});
        }
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
        const std::uint32_t updates=std::clamp(
            options.probe_updates_per_frame,1u,radiance_probe_count);
        const radiance_constants push{
            std::uint32_t(current.index),0u,updates,
            std::clamp(options.rays_per_probe,1u,16u)
        };
        radiance_pipeline.dispatch(cmd,(updates+63u)/64u,1,1,
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
                            vk::PipelineStageFlagBits::eComputeShader |
                            vk::PipelineStageFlagBits::eFragmentShader,
                            {},{},{barrier},{});
    }
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
    // The demo is intentionally self-contained. Until arbitrary GPU mesh
    // layouts have a defined contract, do not silently ignore live scenes.
    if(value.vertices || value.indices || value.instances || value.materials ||
       value.vertex_count || value.index_count || value.instance_count)
        throw std::logic_error("VulkanLumen external scene format not implemented");
    if(impl_)impl_->geometry=value;
}
void system::set_lighting(const lighting& value) {
    if(value.lights || value.count)
        throw std::logic_error("VulkanLumen external light format not implemented");
    if(impl_)impl_->lights=value;
}
void system::set_gbuffer(const gbuffer& value) {
    if(value.depth || value.normal || value.albedo || value.roughness ||
       value.motion)
        throw std::logic_error("VulkanLumen external G-buffer format not implemented");
    if(impl_)impl_->surfaces=value;
}
void system::invalidate() {
    if(impl_)
    {
        impl_->cache_dirty=true;
        impl_->lighting_initialized=false;
        impl_->screen_initialized=false;
        impl_->filter_initialized=false;
        // Reinitialize all cached irradiance on the next record.
        impl_->radiance_initialized=false;
    }
}
void system::record(vk::CommandBuffer cmd,const frame& current) {
    if(!impl_ || !impl_->ready)return;
    impl_->begin_frame(cmd);
    impl_->update_surface_cache(cmd,current);
    impl_->update_surface_lighting(cmd,current);
    impl_->stamp(cmd,1);
    impl_->update_radiance_cache(cmd,current);
    impl_->stamp(cmd,2);
    impl_->draw_scene(cmd,current);
    impl_->stamp(cmd,3);
    impl_->update_screen_probes(cmd,current);
    impl_->filter_screen_probes(cmd);
    impl_->stamp(cmd,4);
}
void system::compose(vk::CommandBuffer cmd) {
    if(!impl_ || !impl_->ready)return;
    impl_->stamp(cmd,5);
    impl_->composite(cmd);
    impl_->stamp(cmd,6);
}
outputs system::result() const noexcept {return {};}
performance system::gpu_performance() const noexcept {
    return impl_?impl_->gpu_stats:performance{};
}
bool system::implemented() const noexcept {
    return impl_ && impl_->ready;
}
const char* system::status() const noexcept {
    return implemented()?
        "lit cards + filtered diffuse GI and rough specular from screen radiance":
        "not initialized";
}
void system::reset() noexcept {impl_.reset();}

} // namespace lumen
