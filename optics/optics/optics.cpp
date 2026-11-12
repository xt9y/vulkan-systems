#include "optics.hpp"
#include <bvh/stackless.hpp>
#include <stdexcept>
#include <algorithm>
#include <array>
#include <gpu/upload.hpp>
#include <unordered_map>
#include <cstdint>
#include <limits>
#include <cstring>
#include <cmath>
#include <vector>
#include <cstdlib>
#include <iostream>

namespace optics {
namespace {
using vec3=vk::util::math::vec3;
using f4=std::array<float,4>;
f4 pack(vec3 value,float w=0.f) {return {value.x,value.y,value.z,w};}
struct alignas(16) gpu_face {f4 origin,u,v,normal,albedo,emission,metric;};
static_assert(sizeof(gpu_face)==112);
}
struct compositor::acceleration {
    struct frame {
        gpu::buffer faces,nodes;
        std::vector<gpu_face> faces_cpu;
        std::vector<bvh::stackless_node> nodes_cpu;
        std::vector<f4> triangles;
    };
    gpu::buffer faces,nodes,face_ids,scene_info;
    std::vector<gpu_face> faces_cpu;
    std::vector<bvh::stackless_node> nodes_cpu;
    std::vector<f4> triangles;
    std::vector<std::uint32_t> first_triangle;
    std::vector<std::unique_ptr<frame>> frames;
};
compositor::~compositor(){reset();}
void compositor::create(vk::RenderPass pass,const std::vector<input>& frames,
                        const char* vertex,const char* fragment,
                        const char* mirror_fragment,
                        const char* planar_fragment)
{
    if(!pass || frames.empty() || !vertex || !fragment ||
       !mirror_fragment || !planar_fragment)
        throw std::invalid_argument("optics: missing presentation pass, frames or shaders");
    reset();
    try {
        std::vector<vk::DescriptorSetLayoutBinding> bindings(18);
        for(std::uint32_t i=0;i<18;++i) {
            bindings[i].binding=i;
            bindings[i].descriptorCount=1;
            bindings[i].descriptorType=(i>=6 && i<=11) || i==13 || i==15 ?
                vk::DescriptorType::eStorageBuffer :
                vk::DescriptorType::eCombinedImageSampler;
            bindings[i].stageFlags=vk::ShaderStageFlagBits::eFragment |
                                    vk::ShaderStageFlagBits::eCompute;
        }
        descriptors_.create(bindings,static_cast<std::uint32_t>(frames.size()));
        const auto host=vk::MemoryPropertyFlagBits::eHostVisible |
                        vk::MemoryPropertyFlagBits::eHostCoherent;
        empty_.create(128,vk::BufferUsageFlagBits::eStorageBuffer,host);
        const std::array<std::uint8_t,128> zeros{};
        empty_.upload(zeros.data(),zeros.size());
        lights_.create(16+16*32,vk::BufferUsageFlagBits::eStorageBuffer,host);
        const std::array<std::uint8_t,16+16*32> empty_lights{};
        lights_.upload(empty_lights.data(),empty_lights.size());
        face_uv_.create(48,vk::BufferUsageFlagBits::eStorageBuffer,host);
        const std::array<std::uint8_t,48> empty_uv{};
        face_uv_.upload(empty_uv.data(),empty_uv.size());
        face_normals_.create(3u*sizeof(std::array<float,4>),
            vk::BufferUsageFlagBits::eStorageBuffer,host);
        probe_space_.create(16,vk::BufferUsageFlagBits::eStorageBuffer,host);
        const std::array<float,4> empty_probe{};
        probe_space_.upload(empty_probe.data(),16);
        const std::array<float,12> empty_normals{};
        face_normals_.upload(empty_normals.data(),sizeof(empty_normals));
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
        // The material resolve replaces the rasterized surface. Alpha
        // blending produced doubled images and unstable mirror edges.
        vk::PipelineColorBlendAttachmentState blend{};
        blend.blendEnable=false;
        blend.colorWriteMask=vk::ColorComponentFlagBits::eR |
            vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB;
        raster.blend_attachments.push_back(blend);
        pipeline_.create(raster,{
            {vk::ShaderStageFlagBits::eVertex,vertex},
            {vk::ShaderStageFlagBits::eFragment,fragment}
        });
        mirror_pipeline_.create(raster,{
            {vk::ShaderStageFlagBits::eVertex,vertex},
            {vk::ShaderStageFlagBits::eFragment,mirror_fragment}
        });
        planar_pipeline_.create(raster,{
            {vk::ShaderStageFlagBits::eVertex,vertex},
            {vk::ShaderStageFlagBits::eFragment,planar_fragment}
        });
        for(std::uint32_t i=0;i<frames.size();++i) {
            const auto& f=frames[i];
            const std::array<vk::ImageView,6> views{
                f.lit,f.opaque_depth,f.normal,f.tint,f.material,f.material_depth
            };
            for(std::uint32_t binding=6;binding<=9;++binding)
                descriptors_.write_buffer(i,binding,empty_.descriptor());
            descriptors_.write_buffer(i,10,lights_.descriptor());
            descriptors_.write_buffer(i,11,face_uv_.descriptor());
            descriptors_.write_buffer(i,13,face_normals_.descriptor());
            descriptors_.write_buffer(i,15,probe_space_.descriptor());
            if(!f.probes)throw std::invalid_argument("optics: missing PCloud probe view");
            vk::DescriptorImageInfo probes{};
            probes.sampler=sampler_;
            probes.imageView=f.probes;
            probes.imageLayout=vk::ImageLayout::eGeneral;
            descriptors_.write_image(i,14,probes);
            if(!f.hdr)throw std::invalid_argument("optics: missing linear HDR view");
            vk::DescriptorImageInfo hdr{};
            hdr.sampler=sampler_;
            hdr.imageView=f.hdr;
            hdr.imageLayout=vk::ImageLayout::eGeneral;
            descriptors_.write_image(i,16,hdr);
            if(!f.reflection)throw std::invalid_argument("optics: missing planar reflection image");
            vk::DescriptorImageInfo mirror{};
            mirror.sampler=sampler_;
            mirror.imageView=f.reflection;
            mirror.imageLayout=vk::ImageLayout::eGeneral;
            descriptors_.write_image(i,17,mirror);
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
        set_textures({},{});
    }catch(...) {reset();throw;}
}
void compositor::set_scene(const std::vector<surface>& source,
                           const std::vector<light>& lights,bool contains_glass)
{
    if(!frame_count_)throw std::logic_error("optics: compositor not initialized");
    if(source.size()>100000u)
        throw std::invalid_argument("optics: too many triangles");
    auto next=std::make_unique<acceleration>();
    next->faces_cpu.reserve(source.size());
    next->triangles.reserve(source.size()*3);
    for(std::uint32_t i=0;i<source.size();++i) {
        const auto& s=source[i];
        const float a=vk::util::math::dot(s.u,s.u);
        const float b=vk::util::math::dot(s.u,s.v);
        const float c=vk::util::math::dot(s.v,s.v);
        const float det=a*c-b*b;
        if(!(det>1e-9f) || !std::isfinite(det) ||
           s.optical_type>2 || s.pattern>2 ||
           !std::isfinite(s.optical_strength) ||
           s.optical_strength<0.f || s.optical_strength>1.f ||
           !std::isfinite(s.roughness) ||
           s.roughness<0.f || s.roughness>1.f ||
           !std::isfinite(s.ior) || s.ior<=1.f)
            throw std::invalid_argument("optics: invalid reflection face");
        const auto n=vk::util::math::normalize(vk::util::math::cross(s.u,s.v));
        const float pattern=s.optical_type ? 0.f :
            s.pattern==1?.25f:s.pattern==2?.5f:0.f;
        next->faces_cpu.push_back({
            pack(s.origin),pack(s.u),pack(s.v),pack(n,1.f),
            pack(s.albedo,s.optical_type==1?s.optical_strength:0.f),
            pack(s.emission,s.optical_type==1?s.ior:
                 s.optical_type==2?-s.optical_strength:pattern),
            {c/det,-b/det,a/det,s.roughness}
        });
        next->triangles.push_back(pack(s.origin,float(i)));
        next->triangles.push_back(pack(s.origin+s.u));
        next->triangles.push_back(pack(s.origin+s.v));
    }
    if(!source.empty()) {
        next->nodes_cpu=bvh::build_stackless(next->triangles);
        if(next->nodes_cpu.empty())
            throw std::runtime_error("optics: BVH construction failed");
    }
    next->first_triangle.assign(source.size(),UINT32_MAX);
    std::vector<std::uint32_t> ids;
    ids.reserve(source.size());
    for(std::size_t i=0;i<next->triangles.size();i+=3) {
        const auto id=std::uint32_t(next->triangles[i][3]);
        ids.push_back(id);
        next->first_triangle[id]=std::uint32_t(i/3);
    }
    const auto host=vk::MemoryPropertyFlagBits::eHostVisible |
                    vk::MemoryPropertyFlagBits::eHostCoherent;
    const auto upload=[&](gpu::buffer& buffer,const void* bytes,std::size_t size) {
        buffer.create(std::max<std::size_t>(size,16u),
                      vk::BufferUsageFlagBits::eStorageBuffer,host);
        if(size)buffer.upload(bytes,size);
    };
    if(!source.empty()) {
        upload(next->faces,next->faces_cpu.data(),
               next->faces_cpu.size()*sizeof(gpu_face));
        upload(next->nodes,next->nodes_cpu.data(),
               next->nodes_cpu.size()*sizeof(bvh::stackless_node));
        upload(next->face_ids,ids.data(),ids.size()*sizeof(std::uint32_t));
        const std::array<std::uint32_t,4> desc{
            std::uint32_t(source.size()),UINT32_MAX,0,0
        };
        upload(next->scene_info,desc.data(),sizeof(desc));
    }
    face_count_=std::uint32_t(source.size());
    contains_glass_=contains_glass || source.empty();
    acceleration_=std::move(next);
    for(std::uint32_t frame=0;frame<frame_count_;++frame) {
        const std::array<vk::DescriptorBufferInfo,4> buffers{
            face_count_?acceleration_->faces.descriptor():empty_.descriptor(),
            face_count_?acceleration_->nodes.descriptor():empty_.descriptor(),
            face_count_?acceleration_->face_ids.descriptor():empty_.descriptor(),
            face_count_?acceleration_->scene_info.descriptor():empty_.descriptor()
        };
        for(std::uint32_t j=0;j<buffers.size();++j)
            descriptors_.write_buffer(frame,6+j,buffers[j]);
    }
    if(face_count_)std::clog<<"optics: BVH faces="<<face_count_
                            <<" with PCloud GI\n";
    set_lights(lights);
}
void compositor::set_lights(const std::vector<light>& lights)
{
    if(!frame_count_)throw std::logic_error("optics: compositor not initialized");
    struct alignas(16) gpu_light {
        std::array<float,4> position_radius{},color_strength{};
    };
    struct alignas(16) gpu_lights {
        std::array<std::uint32_t,4> count{};
        std::array<gpu_light,16> values{};
    };
    static_assert(sizeof(gpu_lights)==16+16*32);
    gpu_lights packet{};
    for(const auto& l:lights) {
        if(packet.count[0]==16)break;
        if(!(std::isfinite(l.position.x) && std::isfinite(l.position.y) &&
             std::isfinite(l.position.z) && std::isfinite(l.color.x) &&
             std::isfinite(l.color.y) && std::isfinite(l.color.z) &&
             std::isfinite(l.radius) && std::isfinite(l.strength)) ||
             l.radius<=0.f || l.strength<=0.f)continue;
        auto& x=packet.values[packet.count[0]++];
        x.position_radius={l.position.x,l.position.y,l.position.z,l.radius};
        x.color_strength={l.color.x,l.color.y,l.color.z,l.strength};
    }
    lights_.upload(&packet,sizeof(packet));
}

void compositor::update_geometry_frame(
    std::uint32_t frame,const std::vector<geometry_update>& updates)
{
    if(frame>=frame_count_ || !acceleration_ || !face_count_)
        throw std::invalid_argument("optics: invalid animated geometry");
    if(updates.empty())return;
    auto& frames=acceleration_->frames;
    if(frames.size()<=frame)frames.resize(frame+1);
    auto& slot=frames[frame];
    if(!slot) {
        slot=std::make_unique<acceleration::frame>();
        slot->faces_cpu=acceleration_->faces_cpu;
        slot->nodes_cpu=acceleration_->nodes_cpu;
        slot->triangles=acceleration_->triangles;
        const auto host=vk::MemoryPropertyFlagBits::eHostVisible |
                        vk::MemoryPropertyFlagBits::eHostCoherent;
        slot->faces.create(slot->faces_cpu.size()*sizeof(gpu_face),
                           vk::BufferUsageFlagBits::eStorageBuffer,host);
        slot->nodes.create(slot->nodes_cpu.size()*sizeof(bvh::stackless_node),
                           vk::BufferUsageFlagBits::eStorageBuffer,host);
    }
    for(const auto& edit:updates) {
        if(edit.face>=face_count_ ||
           acceleration_->first_triangle[edit.face]==UINT32_MAX)
            throw std::out_of_range("optics: skinned face out of range");
        const auto& s=edit.value;
        const float a=vk::util::math::dot(s.u,s.u);
        const float b=vk::util::math::dot(s.u,s.v);
        const float c=vk::util::math::dot(s.v,s.v);
        const float det=a*c-b*b;
        if(!(det>1e-10f) || !std::isfinite(det))continue;
        auto& face=slot->faces_cpu[edit.face];
        face.origin=pack(s.origin);
        face.u=pack(s.u);
        face.v=pack(s.v);
        face.normal=pack(vk::util::math::normalize(
            vk::util::math::cross(s.u,s.v)),1.f);
        face.metric[0]=c/det;
        face.metric[1]=-b/det;
        face.metric[2]=a/det;
        const auto first=std::size_t(acceleration_->first_triangle[edit.face])*3;
        slot->triangles[first]=pack(s.origin,float(edit.face));
        slot->triangles[first+1]=pack(s.origin+s.u);
        slot->triangles[first+2]=pack(s.origin+s.v);
    }
    // Refit the existing SAH topology, from leaves to parents.
    for(std::size_t j=slot->nodes_cpu.size();j-->0;) {
        auto& node=slot->nodes_cpu[j];
        const auto inf=std::numeric_limits<float>::infinity();
        vec3 lo{inf,inf,inf},hi{-inf,-inf,-inf};
        const auto add=[&](vec3 p) {
            lo={std::min(lo.x,p.x),std::min(lo.y,p.y),std::min(lo.z,p.z)};
            hi={std::max(hi.x,p.x),std::max(hi.y,p.y),std::max(hi.z,p.z)};
        };
        const auto count=node.first_and_count&15u;
        if(count) {
            const auto first=node.first_and_count>>4u;
            for(std::uint32_t k=0;k<count;++k)
                for(std::uint32_t v=0;v<3;++v) {
                    const auto& p=slot->triangles[(first+k)*3u+v];
                    add({p[0],p[1],p[2]});
                }
        } else {
            for(std::uint32_t child=std::uint32_t(j)+1u;
                child<node.next;child=slot->nodes_cpu[child].next) {
                const auto& sub=slot->nodes_cpu[child];
                add({sub.minimum[0],sub.minimum[1],sub.minimum[2]});
                add({sub.maximum[0],sub.maximum[1],sub.maximum[2]});
            }
        }
        if(!std::isfinite(lo.x))continue;
        node.minimum={lo.x-.0001f,lo.y-.0001f,lo.z-.0001f};
        node.maximum={hi.x+.0001f,hi.y+.0001f,hi.z+.0001f};
    }
    slot->faces.upload(slot->faces_cpu.data(),
                       slot->faces_cpu.size()*sizeof(gpu_face));
    slot->nodes.upload(slot->nodes_cpu.data(),
                       slot->nodes_cpu.size()*sizeof(bvh::stackless_node));
    descriptors_.write_buffer(frame,6,slot->faces.descriptor());
    descriptors_.write_buffer(frame,7,slot->nodes.descriptor());
}
void compositor::update_normals(
    std::uint32_t frame,const std::vector<face_normal_update>& updates)
{
    if(frame>=frame_count_ || original_normals_.size()!=face_count_)
        throw std::invalid_argument("optics: invalid animated normals");
    if(updates.empty())return;
    if(frame_normals_.size()<=frame)frame_normals_.resize(frame+1);
    auto& state=frame_normals_[frame];
    if(!state) {
        state=std::make_unique<animated_normals>();
        state->faces=original_normals_;
        const auto memory=vk::MemoryPropertyFlagBits::eHostVisible |
                          vk::MemoryPropertyFlagBits::eHostCoherent;
        state->buffer.create(state->faces.size()*sizeof(face_normals),
                             vk::BufferUsageFlagBits::eStorageBuffer,memory);
    }
    for(const auto& change:updates) {
        if(change.face>=state->faces.size())
            throw std::out_of_range("optics: skinned face out of range");
        state->faces[change.face]=change.value;
    }
    state->buffer.upload(state->faces.data(),
                         state->faces.size()*sizeof(face_normals));
    descriptors_.write_buffer(frame,13,state->buffer.descriptor());
}

void compositor::set_probe_space(const std::array<float,4>& space)
{
    probe_space_.upload(space.data(),sizeof(space));
}
void compositor::set_probe_views(const std::vector<vk::ImageView>& views)
{
    if(views.size()!=frame_count_)
        throw std::invalid_argument("optics: probe view count mismatch");
    for(std::uint32_t i=0;i<frame_count_;++i) {
        if(!views[i])throw std::invalid_argument("optics: missing probe image");
        vk::DescriptorImageInfo probe{};
        probe.sampler=sampler_;
        probe.imageView=views[i];
        probe.imageLayout=vk::ImageLayout::eGeneral;
        descriptors_.write_image(i,14,probe);
    }
}
void compositor::set_normals(const std::vector<face_normals>& normals)
{
    original_normals_=normals;
    frame_normals_.clear();
    if(!frame_count_ || normals.size()!=face_count_)
        throw std::invalid_argument("optics: face normals are not BVH aligned");
    // Like set_scene(), this is called only with previous GPU work idle.
    static_assert(sizeof(face_normals)==48);
    const auto host=vk::MemoryPropertyFlagBits::eHostVisible |
                    vk::MemoryPropertyFlagBits::eHostCoherent;
    face_normals_.reset();
    face_normals_.create(std::max<std::size_t>(1,normals.size())*
                         sizeof(face_normals),
                         vk::BufferUsageFlagBits::eStorageBuffer,host);
    const face_normals empty{};
    face_normals_.upload(normals.empty()?&empty:normals.data(),
                         std::max<std::size_t>(1,normals.size())*
                         sizeof(face_normals));
    for(std::uint32_t frame=0;frame<frame_count_;++frame)
        descriptors_.write_buffer(frame,13,face_normals_.descriptor());
}

void compositor::set_textures(const std::vector<face_uv>& faces,
                              const std::vector<texture_rgba>& textures)
{
    if(!frame_count_)throw std::logic_error("optics: compositor not initialized");
    // This function must be called while all submissions using the old
    // atlas/descriptors have completed, as with set_scene().
    struct placement {std::uint32_t id,x,y,w,h;};
    std::vector<texture_rgba> inputs;
    inputs.reserve(textures.size()+1);
    inputs.push_back({0,1,1,{255,255,255,255}});
    for(const auto& texture:textures) {
        if(!texture.id)continue;
        if(!texture.width || !texture.height ||
           texture.width>8190 || texture.height>8190 ||
           std::uint64_t(texture.width)*texture.height*4u!=texture.pixels.size())
            throw std::invalid_argument("optics: invalid reflection texture");
        inputs.push_back(texture);
    }
    std::uint32_t width=1;
    for(const auto& x:inputs)
        while(width<x.width+2)width*=2;
    std::vector<placement> placed;
    std::uint32_t height=0;
    for(;width<=8192;width*=2) {
        placed.clear();
        std::uint32_t x=0,y=0,row=0;
        bool fits=true;
        for(const auto& texture:inputs) {
            if(x+texture.width+2>width) {x=0;y+=row;row=0;}
            if(y+texture.height+2>8192) {fits=false;break;}
            placed.push_back({texture.id,x,y,texture.width,texture.height});
            x+=texture.width+2;
            row=std::max(row,texture.height+2);
        }
        if(fits) {
            height=1;
            while(height<y+row)height*=2;
            break;
        }
    }
    if(!height || height>8192 || width>8192)
        throw std::length_error("optics: reflection texture atlas exceeds 8192 pixels");
    std::vector<std::uint8_t> pixels(std::size_t(width)*height*4,255);
    std::unordered_map<std::uint32_t,placement> locations;
    for(std::size_t i=0;i<inputs.size();++i) {
        const auto& t=inputs[i];
        const auto& r=placed[i];
        if(!locations.emplace(t.id,r).second)
            throw std::invalid_argument("optics: duplicate scene texture ID");
        // Duplicate the edge pixels into a 1-pixel gutter. Filtering cannot
        // leak colors from neighboring atlas entries.
        for(int py=-1;py<=int(t.height);++py)
        for(int px=-1;px<=int(t.width);++px) {
            int sx=std::clamp(px,0,int(t.width)-1);
            int sy=std::clamp(py,0,int(t.height)-1);
            std::size_t dst=(std::size_t(r.y+1+py)*width+
                             std::size_t(r.x+1+px))*4;
            std::size_t src=(std::size_t(sy)*t.width+std::size_t(sx))*4;
            std::memcpy(pixels.data()+dst,t.pixels.data()+src,4);
        }
    }
    struct alignas(16) gpu_uv {
        std::array<float,4> uv01{},uv2_origin{},extent{};
        std::array<float,4> metallic_rect{},normal_rect{};
        std::array<float,4> occlusion_rect{},emissive_rect{};
    };
    static_assert(sizeof(gpu_uv)==112);
    std::vector<gpu_uv> mapping(std::max<std::size_t>(faces.size(),1));
    const auto rect_for=[&](std::uint32_t id) -> std::array<float,4> {
        if(!id)return {};
        const auto it=locations.find(id);
        if(it==locations.end())
            throw std::invalid_argument("optics: reflection face references missing PBR texture");
        const auto& r=it->second;
        return {float(r.x+1)/float(width),float(r.y+1)/float(height),
                float(r.w)/float(width),float(r.h)/float(height)};
    };
    for(std::size_t i=0;i<faces.size();++i) {
        const auto& f=faces[i];
        auto& m=mapping[i];
        m.uv01={f.a[0],f.a[1],f.b[0],f.b[1]};
        m.uv2_origin[0]=f.c[0];
        m.uv2_origin[1]=f.c[1];
        m.extent[2]=std::clamp(f.base_alpha,0.f,1.f);
        m.extent[3]=std::clamp(f.alpha_cutoff,0.f,1.f);
        if(f.texture_id) {
            const auto base=rect_for(f.texture_id);
            m.uv2_origin[2]=base[0];
            m.uv2_origin[3]=base[1];
            m.extent[0]=base[2];
            m.extent[1]=base[3];
        }
        m.metallic_rect=rect_for(f.metallic_roughness_texture);
        m.normal_rect=rect_for(f.normal_texture);
        m.occlusion_rect=rect_for(f.occlusion_texture);
        m.emissive_rect=rect_for(f.emissive_texture);
    }
    const auto host=vk::MemoryPropertyFlagBits::eHostVisible |
                    vk::MemoryPropertyFlagBits::eHostCoherent;
    face_uv_.reset();
    face_uv_.create(mapping.size()*sizeof(gpu_uv),
                    vk::BufferUsageFlagBits::eStorageBuffer,host);
    face_uv_.upload(mapping.data(),mapping.size()*sizeof(gpu_uv));
    gpu::upload::batch upload;
    auto image=upload.add_image(pixels.data(),width,height);
    upload.finish();
    gpu::upload::retire(true);
    if(atlas_view_)vk::image_view::destroy(atlas_view_);
    atlas_view_=nullptr;
    atlas_.reset();
    atlas_=image;
    vk::ImageViewCreateInfo info{};
    info.image=atlas_.handle;
    info.viewType=vk::ImageViewType::e2D;
    info.format=vk::Format::eR8G8B8A8Unorm;
    info.subresourceRange={vk::ImageAspectFlagBits::eColor,0,1,0,1};
    atlas_view_=vk::image_view::create(info);
    if(!atlas_sampler_) {
        vk::SamplerCreateInfo sampler{};
        sampler.magFilter=vk::Filter::eLinear;
        sampler.minFilter=vk::Filter::eLinear;
        sampler.mipmapMode=vk::SamplerMipmapMode::eNearest;
        sampler.addressModeU=sampler.addressModeV=sampler.addressModeW=
            vk::SamplerAddressMode::eClampToEdge;
        atlas_sampler_=vk::device::get()->handle.createSampler(sampler);
    }
    vk::DescriptorImageInfo atlas{};
    atlas.sampler=atlas_sampler_;
    atlas.imageView=atlas_view_;
    atlas.imageLayout=vk::ImageLayout::eShaderReadOnlyOptimal;
    for(std::uint32_t i=0;i<frame_count_;++i) {
        descriptors_.write_buffer(i,11,face_uv_.descriptor());
        descriptors_.write_image(i,12,atlas);
    }
}

void compositor::record(vk::CommandBuffer cmd,std::uint32_t frame,
                        const camera& view,const std::vector<vk::Rect2D>& regions,
                        vk::Extent2D extent) const
{
    if(!cmd || !pipeline_.handle() || !mirror_pipeline_.handle() ||
       !planar_pipeline_.handle() || frame>=frame_count_)
        throw std::logic_error("optics: missing pipeline or invalid frame");
    if(regions.empty())return;
    // A/B control: compare the exact same mirror scene with the original
    // general shader, without changing geometry or visual settings.
    static const bool force_general=
        std::getenv("OPTICS_PROFILE_GENERAL")!=nullptr;
    cmd.bindPipeline(vk::PipelineBindPoint::eGraphics,
                     (view.eye.w>1.5f && view.eye.w<2.5f)?planar_pipeline_.handle():
                     (contains_glass_||force_general)?
                     pipeline_.handle():mirror_pipeline_.handle());
    const vk::DescriptorSet set=descriptors_.set(frame);
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,layout_,0,{set},{});
    cmd.pushConstants(layout_,vk::ShaderStageFlagBits::eFragment,0,sizeof(camera),&view);
    // Pipeline viewport and scissor are dynamic. Present only the visible
    // optical material regions, then restore state for the editor overlay.
    for(const auto& rectangle:regions) {
        if(!rectangle.extent.width || !rectangle.extent.height)continue;
        cmd.setScissor(0,{rectangle});
        cmd.draw(3,1,0,0);
    }
    cmd.setScissor(0,{vk::Rect2D{{0,0},extent}});
}
void compositor::reset() noexcept
{
    planar_pipeline_.reset();
    mirror_pipeline_.reset();
    pipeline_.reset();
    contains_glass_=true;
    if(layout_)vk::device::get()->handle.destroyPipelineLayout(layout_);
    layout_=nullptr;
    descriptors_.reset();
    acceleration_.reset();
    empty_.reset();
    lights_.reset();
    face_uv_.reset();
    frame_normals_.clear();
    original_normals_.clear();
    face_normals_.reset();
    probe_space_.reset();
    face_count_=0;
    if(atlas_view_)vk::image_view::destroy(atlas_view_);
    atlas_view_=nullptr;
    atlas_.reset();
    if(atlas_sampler_)vk::device::get()->handle.destroySampler(atlas_sampler_);
    atlas_sampler_=nullptr;
    if(sampler_)vk::device::get()->handle.destroySampler(sampler_);
    sampler_=nullptr;
    frame_count_=0;
}
}
