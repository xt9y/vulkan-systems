#include "pmap.hpp"
#include <gpu/buffer.hpp>
#include <gpu/image.hpp>
#include <gpu/descriptors.hpp>
#include <gpu/compute.hpp>
#include <bvh/stackless.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace pm
{
    namespace
    {
        using vec4=vk::util::math::vec4;
        constexpr std::uint32_t cells=16u*16u*16u,photon_bytes=48u;
        using f4=std::array<float,4>;
        f4 pack(vec3 v,float w=0.f){return {{v.x,v.y,v.z,w}};}
        float squared(vec3 v){return v.x*v.x+v.y*v.y+v.z*v.z;}
        vec3 min3(vec3 a,vec3 b)
        {
            return {std::min(a.x,b.x),std::min(a.y,b.y),std::min(a.z,b.z)};
        }
        vec3 max3(vec3 a,vec3 b)
        {
            return {std::max(a.x,b.x),std::max(a.y,b.y),std::max(a.z,b.z)};
        }
        struct alignas(16) gpu_face
        {
            f4 origin,u,v,normal,albedo,emission,metric;
        };
        static_assert(sizeof(gpu_face)==112,"GLSL Face layout mismatch");
        struct alignas(16) scene_parameters
        {
            std::array<std::uint32_t,4> description{};
            f4 grid{};
            std::array<std::uint32_t,4> extra{};
        };
        static_assert(sizeof(scene_parameters)==48,"GLSL SceneInfo mismatch");
        struct alignas(16) camera_push
        {
            f4 eye{},right{},up{};
            std::array<std::uint32_t,4> params{},config{};
            f4 settings{},sphere{},material{};
        };
        static_assert(sizeof(camera_push)==128,"portable push constant limit");
        struct acceleration
        {
            std::vector<gpu_face> faces;
            std::vector<bvh::stackless_node> nodes;
            std::vector<f4> triangles;
            std::vector<std::array<std::uint32_t,4>> cutouts;
            scene_parameters parameters{};
        };
        acceleration make_acceleration(const scene& source,float photon_radius,
                                       bool transport_only)
        {
            if(source.surfaces.empty() || source.surfaces.size()>100000u ||
               source.emitter>=source.surfaces.size())
                throw std::invalid_argument("pm: invalid scene or emitter");
            acceleration result;
            result.faces.reserve(source.surfaces.size());
            const auto inf=std::numeric_limits<float>::infinity();
            vec3 low{inf,inf,inf},high{-inf,-inf,-inf};
            for(std::uint32_t i=0;i<source.surfaces.size();++i)
            {
                const surface& f=source.surfaces[i];
                const float a=squared(f.u),b=vk::util::math::dot(f.u,f.v),c=squared(f.v);
                const float determinant=a*c-b*b;
                if(!(determinant>1e-9f) || !std::isfinite(determinant))
                    throw std::invalid_argument("pm: degenerate face");
                const float length=std::sqrt(squared(vk::util::math::cross(f.u,f.v)));
                if(!(squared(f.normal)>.5f) || !std::isfinite(length))
                    throw std::invalid_argument("pm: invalid face normal");
                const vec3 n=vk::util::math::normalize(f.normal);
                const float area=length*(f.triangle?.5f:1.f);
                if(f.optical_type>2 || f.pattern>2 ||
                   !std::isfinite(f.optical_strength) ||
                   f.optical_strength<0.f || f.optical_strength>1.f ||
                   !std::isfinite(f.roughness) || f.roughness<0.f ||
                   f.roughness>1.f || !std::isfinite(f.ior) ||
                   f.ior<=1.f)
                    throw std::invalid_argument("pm: invalid optical material");
                const float transmission=f.optical_type==1 ?
                    f.optical_strength : 0.f;
                const float conductor=f.optical_type==2 ?
                    -f.optical_strength : 0.f;
                // Diffuse surface pattern lives in otherwise unused
                // emission.w: .25 world checker, .50 local checker.
                const float pattern=f.optical_type==0 ?
                    (f.pattern==1?.25f:f.pattern==2?.5f:0.f):0.f;
                result.faces.push_back({
                    pack(f.origin),pack(f.u),pack(f.v),
                    pack(n,f.triangle?1.f:0.f),
                    pack(f.albedo,transmission),
                    pack(f.emission, f.optical_type==1 ? f.ior :
                         f.optical_type==2 ? conductor : pattern),
                    {{c/determinant,-b/determinant,
                      a/determinant,f.optical_type ? f.roughness : area}}
                });
                auto add_tri=[&](vec3 x,vec3 y,vec3 z)
                {
                    // Embed exact source face ID before BVH sorting. Two
                    // coincident triangles can have different material/UVs.
                    const std::array<f4,3> tri{
                        pack(x,float(i)),pack(y),pack(z)
                    };
                    result.triangles.insert(result.triangles.end(),
                                            tri.begin(),tri.end());
                };
                const vec3 p=f.origin,q=p+f.u,r=p+f.v;
                if(f.triangle)add_tri(p,q,r);
                else
                {
                    add_tri(p,q,q+f.v);
                    add_tri(p,q+f.v,r);
                }
                low=min3(low,min3(p,min3(q,r)));
                high=max3(high,max3(p,max3(q,r)));
                if(!f.triangle)
                {
                    low=min3(low,q+f.v);
                    high=max3(high,q+f.v);
                }
            }
            if(!transport_only &&
               squared(source.surfaces[source.emitter].emission)<=0.f)
                throw std::invalid_argument("pm: selected emitter has no emission");
            result.nodes=bvh::build_stackless(result.triangles);
            if(result.nodes.empty())
                throw std::invalid_argument("pm: BVH build failed");
            for(const auto& mask:source.cutouts)
            {
                if(mask.receiver>=source.surfaces.size() ||
                   mask.cover>=source.surfaces.size())
                    throw std::invalid_argument("pm: invalid receiver cutout");
                result.cutouts.push_back({mask.receiver,mask.cover,0,0});
            }
            const vec3 pad{photon_radius*2.f,photon_radius*2.f,photon_radius*2.f};
            low=low-pad;
            high=high+pad;
            const vec3 extent=high-low;
            const float side=std::max({extent.x,extent.y,extent.z});
            const float cell_size=std::max(photon_radius,side/16.f);
            const std::uint32_t count=std::uint32_t(source.surfaces.size());
            const std::uint32_t columns=std::uint32_t(std::ceil(std::sqrt(double(count))));
            const std::uint32_t rows=(count+columns-1)/columns;
            result.parameters.description={count,source.emitter,columns,rows};
            result.parameters.grid={{low.x,low.y,low.z,cell_size}};
            result.parameters.extra[0]=std::uint32_t(result.cutouts.size());
            result.parameters.extra[1]=source.omnidirectional_emitter?1u:0u;
            return result;
        }
        void image_barrier(vk::CommandBuffer command,gpu::image& resource,
                           vk::PipelineStageFlags from,vk::AccessFlags read_from,
                           vk::PipelineStageFlags to,vk::AccessFlags write_to,
                           vk::ImageLayout old_layout,vk::ImageLayout new_layout)
        {
            vk::ImageMemoryBarrier b{};
            b.srcAccessMask=read_from;
            b.dstAccessMask=write_to;
            b.oldLayout=old_layout;
            b.newLayout=new_layout;
            b.srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
            b.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
            b.image=resource.handle();
            b.subresourceRange.aspectMask=vk::ImageAspectFlagBits::eColor;
            b.subresourceRange.levelCount=1;
            b.subresourceRange.layerCount=1;
            command.pipelineBarrier(from,to,{},{},{},{b});
        }
        void compute_barrier(vk::CommandBuffer command)
        {
            vk::MemoryBarrier b{};
            b.srcAccessMask=vk::AccessFlagBits::eShaderWrite;
            b.dstAccessMask=vk::AccessFlagBits::eShaderRead |
                            vk::AccessFlagBits::eShaderWrite;
            command.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                    vk::PipelineStageFlagBits::eComputeShader,
                                    {},{b},{},{});
        }
        void validate(const configuration& cfg,vk::Extent2D extent)
        {
            if(!extent.width || !extent.height || cfg.photons<256 ||
               cfg.photons>65536 || cfg.bounces<2 || cfg.bounces>6 ||
               cfg.photon_radius<.05f || cfg.photon_radius>.5f ||
               cfg.gi_updates<1 || cfg.gi_updates>2048 ||
               cfg.direct_updates<1 || cfg.direct_updates>512 ||
               (cfg.direct_spp!=4 && cfg.direct_spp!=8) ||
               (cfg.atlas_tile!=32 && cfg.atlas_tile!=64 &&
                cfg.atlas_tile!=128) ||
               (cfg.denoise_passes!=1 && cfg.denoise_passes!=2) ||
               (cfg.edge_spp!=0 && cfg.edge_spp!=2 &&
                cfg.edge_spp!=4 && cfg.edge_spp!=8) ||
               (cfg.screen_spp<1 || cfg.screen_spp>8) ||
               cfg.debug_view>6 || cfg.shader_directory.empty())
                throw std::invalid_argument("pm: invalid configuration");
        }
        std::string shader_path(const configuration& cfg,const char* name)
        {
            return cfg.shader_directory+"/pmap_"+name+".spv";
        }
    }

    struct mapper::implementation
    {
        scene world;
        configuration cfg;
        vk::Extent2D extent{},atlas{};
        std::uint32_t photon_capacity=0;
        std::uint32_t gi_samples=0,direct_samples=0,stable_frames=0;
        std::uint64_t frame=0;
        bool prepared=false,indirect_prepared=false,direct_prepared=false;
        bool history_valid=false,previous_valid=false;
        bool have_output=false;
        f4 last_eye{},last_right{},last_up{};
        gpu::buffer faces,raw,counts,counter,sorted,offsets,cursors;
        gpu::buffer nodes,triangles,face_ids,previous_camera,scene_info,cutouts;
        gpu::buffer direct_moments,direct_filtered;
        std::array<gpu::image,2> history,positions;
        gpu::image temporal,scratch,indirect,direct;
        gpu::descriptors sets;
        gpu::compute clear,emit,bin,prefix,scatter,gather_gi;
        gpu::compute direct_pass,direct_filter,gather,temporal_pass,denoise;
        std::uint32_t node_count=0;
        std::uint32_t current=0;
        acceleration base_accel;
        // World-space geometry may be animated without rebuilding pipelines,
        // photon maps, UV atlases or the sorted triangle/BVH topology.
        struct animated_frame {
            gpu::buffer faces,nodes,triangles;
            std::vector<gpu_face> face_data;
            std::vector<bvh::stackless_node> bounds;
            std::vector<f4> vertex_data;
        };
        std::vector<std::unique_ptr<animated_frame>> animated;
        std::vector<std::uint32_t> face_triangle;

        geometry_output refit_geometry(
            std::uint32_t image,const std::vector<geometry_update>& updates)
        {
            if(updates.empty())
                return {faces.handle(),nodes.handle(),triangles.handle(),
                        scene_info.handle(),sorted.handle(),offsets.handle(),
                        face_ids.handle()};
            if(image>=16)throw std::out_of_range("pm: invalid animation frame");
            if(animated.size()<=image)animated.resize(image+1);
            auto& slot=animated[image];
            if(!slot) {
                slot=std::make_unique<animated_frame>();
                slot->face_data=base_accel.faces;
                slot->bounds=base_accel.nodes;
                slot->vertex_data=base_accel.triangles;
                const auto usage=vk::BufferUsageFlagBits::eStorageBuffer;
                const auto memory=vk::MemoryPropertyFlagBits::eHostVisible |
                                  vk::MemoryPropertyFlagBits::eHostCoherent;
                slot->faces.create(slot->face_data.size()*sizeof(gpu_face),
                                   usage,memory);
                slot->nodes.create(slot->bounds.size()*sizeof(bvh::stackless_node),
                                   usage,memory);
                slot->triangles.create(slot->vertex_data.size()*sizeof(f4),
                                       usage,memory);
            }
            for(const auto& edit:updates) {
                if(edit.face>=slot->face_data.size() ||
                   edit.face>=face_triangle.size() ||
                   face_triangle[edit.face]==UINT32_MAX ||
                   !edit.value.triangle)
                    throw std::invalid_argument("pm: invalid skinned face");
                const auto& f=edit.value;
                const float a=squared(f.u),b=vk::util::math::dot(f.u,f.v),
                            c=squared(f.v),det=a*c-b*b;
                if(!(det>1e-10f) || !std::isfinite(det))continue;
                auto& face=slot->face_data[edit.face];
                const vec3 n=vk::util::math::normalize(
                    vk::util::math::cross(f.u,f.v));
                face.origin=pack(f.origin);
                face.u=pack(f.u);
                face.v=pack(f.v);
                face.normal=pack(n,1.f);
                face.metric[0]=c/det;
                face.metric[1]=-b/det;
                face.metric[2]=a/det;
                const auto tri=std::size_t(face_triangle[edit.face])*3u;
                slot->vertex_data[tri]=pack(f.origin,float(edit.face));
                slot->vertex_data[tri+1]=pack(f.origin+f.u);
                slot->vertex_data[tri+2]=pack(f.origin+f.v);
            }
            // Refit leaf bounds and then internal bounds bottom-up.
            // No SAH partitioning, Morton sort or GPU pipeline rebuild.
            for(std::size_t j=slot->bounds.size();j-- >0;) {
                auto& node=slot->bounds[j];
                const float inf=std::numeric_limits<float>::infinity();
                vec3 lo{inf,inf,inf},hi{-inf,-inf,-inf};
                const auto add=[&](vec3 p) {
                    lo=min3(lo,p);
                    hi=max3(hi,p);
                };
                const auto count=node.first_and_count&15u;
                if(count) {
                    const auto first=node.first_and_count>>4u;
                    for(std::uint32_t k=0;k<count;++k)
                        for(std::uint32_t v=0;v<3;++v) {
                            const auto& p=slot->vertex_data[(first+k)*3u+v];
                            add({p[0],p[1],p[2]});
                        }
                } else {
                    // In the stackless topology each child subtree ends at
                    // its escape index, and children are sequential.
                    for(std::uint32_t child=std::uint32_t(j)+1u;
                        child<node.next;child=slot->bounds[child].next) {
                        const auto& sub=slot->bounds[child];
                        add({sub.minimum[0],sub.minimum[1],sub.minimum[2]});
                        add({sub.maximum[0],sub.maximum[1],sub.maximum[2]});
                    }
                }
                if(!std::isfinite(lo.x))continue;
                node.minimum={lo.x-.0001f,lo.y-.0001f,lo.z-.0001f};
                node.maximum={hi.x+.0001f,hi.y+.0001f,hi.z+.0001f};
            }
            slot->faces.upload(slot->face_data.data(),
                               slot->face_data.size()*sizeof(gpu_face));
            slot->nodes.upload(slot->bounds.data(),
                               slot->bounds.size()*sizeof(bvh::stackless_node));
            slot->triangles.upload(slot->vertex_data.data(),
                                   slot->vertex_data.size()*sizeof(f4));
            return {slot->faces.handle(),slot->nodes.handle(),
                    slot->triangles.handle(),scene_info.handle(),
                    sorted.handle(),offsets.handle(),face_ids.handle()};
        }

        void create(scene input,vk::Extent2D dimensions,configuration config)
        {
            validate(config,dimensions);
            acceleration accel=make_acceleration(input,config.photon_radius,
                                                   config.transport_only);
            cfg=std::move(config);
            world=std::move(input);
            extent=cfg.transport_only ? vk::Extent2D{1,1} : dimensions;
            atlas.width=cfg.transport_only ? 1u :
                accel.parameters.description[2]*cfg.atlas_tile;
            atlas.height=cfg.transport_only ? 1u :
                accel.parameters.description[3]*cfg.atlas_tile;
            photon_capacity=cfg.photons*cfg.bounces;
            node_count=std::uint32_t(accel.nodes.size());
            const auto storage=vk::BufferUsageFlagBits::eStorageBuffer;
            const auto local=vk::MemoryPropertyFlagBits::eDeviceLocal;
            const auto coherent=vk::MemoryPropertyFlagBits::eHostVisible |
                                vk::MemoryPropertyFlagBits::eHostCoherent;
            auto upload=[&](gpu::buffer& buffer,const void* data,std::size_t bytes)
            {
                buffer.create(std::max<std::size_t>(bytes,16),storage,coherent);
                if(bytes)buffer.upload(data,bytes);
            };
            upload(faces,accel.faces.data(),accel.faces.size()*sizeof(gpu_face));
            upload(nodes,accel.nodes.data(),accel.nodes.size()*sizeof(bvh::stackless_node));
            upload(triangles,accel.triangles.data(),accel.triangles.size()*sizeof(f4));
            // Optical traversal needs only a face ID per ordered triangle;
            // the photon kernels still use the original triangle positions.
            std::vector<std::uint32_t> compact_ids;
            compact_ids.reserve(accel.triangles.size()/3);
            for(std::size_t i=0;i+2<accel.triangles.size();i+=3)
                compact_ids.push_back(std::uint32_t(accel.triangles[i][3]));
            upload(face_ids,compact_ids.data(),
                   compact_ids.size()*sizeof(std::uint32_t));
            upload(scene_info,&accel.parameters,sizeof(scene_parameters));
            upload(cutouts,accel.cutouts.data(),
                   accel.cutouts.size()*sizeof(std::array<std::uint32_t,4>));
            base_accel=std::move(accel);
            face_triangle.assign(base_accel.faces.size(),UINT32_MAX);
            for(std::size_t i=0;i+2<base_accel.triangles.size();i+=3) {
                const auto face=std::uint32_t(base_accel.triangles[i][3]);
                if(face<face_triangle.size() && face_triangle[face]==UINT32_MAX)
                    face_triangle[face]=std::uint32_t(i/3);
            }
            previous_camera.create(sizeof(f4)*3u,storage,coherent);
            raw.create(vk::DeviceSize(photon_capacity)*photon_bytes,storage,local);
            sorted.create(vk::DeviceSize(photon_capacity)*photon_bytes,storage,local);
            counts.create(cells*sizeof(std::uint32_t),storage,local);
            counter.create(sizeof(std::uint32_t),storage,local);
            offsets.create((cells+1u)*sizeof(std::uint32_t),storage,local);
            cursors.create(cells*sizeof(std::uint32_t),storage,local);
            const auto bytes=vk::DeviceSize(atlas.width)*atlas.height*sizeof(f4);
            direct_moments.create(bytes,storage,local);
            direct_filtered.create(bytes,storage,local);

            vk::ImageCreateInfo desc{};
            desc.imageType=vk::ImageType::e2D;
            desc.format=vk::Format::eR16G16B16A16Sfloat;
            desc.extent.width=extent.width;
            desc.extent.height=extent.height;
            desc.extent.depth=1;
            desc.mipLevels=desc.arrayLayers=1;
            desc.samples=vk::SampleCountFlagBits::e1;
            desc.tiling=vk::ImageTiling::eOptimal;
            desc.usage=vk::ImageUsageFlagBits::eStorage;
            desc.sharingMode=vk::SharingMode::eExclusive;
            vk::ImageViewCreateInfo view{};
            view.viewType=vk::ImageViewType::e2D;
            view.format=desc.format;
            view.subresourceRange.aspectMask=vk::ImageAspectFlagBits::eColor;
            view.subresourceRange.levelCount=view.subresourceRange.layerCount=1;
            auto make_image=[&](gpu::image& out)
            {
                out.create(desc,local);
                out.make_view(view);
            };
            for(auto& x:history)make_image(x);
            for(auto& x:positions)make_image(x);
            make_image(temporal);
            make_image(scratch);
            desc.format=vk::Format::eR32G32B32A32Sfloat;
            desc.extent.width=atlas.width;
            desc.extent.height=atlas.height;
            view.format=desc.format;
            make_image(indirect);
            make_image(direct);

            std::vector<vk::DescriptorSetLayoutBinding> bindings;
            for(std::uint32_t i=0;i<=24u;++i)
            {
                if(i==11 || i==13 || i==15)continue;
                vk::DescriptorSetLayoutBinding b{};
                b.binding=i;
                b.descriptorType=(i==0 || i==8 || i==12 || i==14 ||
                                  i==16 || i==17 || i==18 || i==19)?
                                 vk::DescriptorType::eStorageImage:
                                 vk::DescriptorType::eStorageBuffer;
                b.descriptorCount=1;
                b.stageFlags=vk::ShaderStageFlagBits::eCompute;
                bindings.push_back(b);
            }
            sets.create(std::move(bindings),2);
            auto image_descriptor=[&](std::uint32_t set,std::uint32_t binding,
                                      const gpu::image& value)
            {
                vk::DescriptorImageInfo info{};
                info.imageView=value.view();
                info.imageLayout=vk::ImageLayout::eGeneral;
                sets.write_image(set,binding,info);
            };
            for(std::uint32_t set=0;set<2;++set)
            {
                image_descriptor(set,0,history[set]);
                sets.write_buffer(set,1,faces.descriptor());
                sets.write_buffer(set,2,raw.descriptor());
                sets.write_buffer(set,3,counts.descriptor());
                sets.write_buffer(set,4,counter.descriptor());
                sets.write_buffer(set,5,sorted.descriptor());
                sets.write_buffer(set,6,offsets.descriptor());
                sets.write_buffer(set,7,cursors.descriptor());
                image_descriptor(set,8,indirect);
                sets.write_buffer(set,9,nodes.descriptor());
                sets.write_buffer(set,10,triangles.descriptor());
                image_descriptor(set,12,positions[set]);
                image_descriptor(set,14,positions[set^1u]);
                image_descriptor(set,16,history[set^1u]);
                image_descriptor(set,17,temporal);
                image_descriptor(set,18,scratch);
                image_descriptor(set,19,direct);
                sets.write_buffer(set,20,previous_camera.descriptor());
                sets.write_buffer(set,21,direct_moments.descriptor());
                sets.write_buffer(set,22,direct_filtered.descriptor());
                sets.write_buffer(set,23,scene_info.descriptor());
                sets.write_buffer(set,24,cutouts.descriptor());
            }
            const auto layout=sets.layout();
            const auto pipeline=[&](gpu::compute& out,const char* name,bool push)
            {
                out.create(shader_path(cfg,name),{layout},
                           push?sizeof(camera_push):0);
            };
            pipeline(clear,"photon_clear.comp",false);
            pipeline(emit,"photon_emit.comp",true);
            pipeline(bin,"photon_bin.comp",true);
            pipeline(prefix,"photon_prefix.comp",false);
            pipeline(scatter,"photon_scatter.comp",true);
            if(!cfg.transport_only) {
                pipeline(gather_gi,"photon_cache.comp",true);
                pipeline(direct_pass,"photon_direct.comp",true);
                pipeline(direct_filter,"photon_direct_filter.comp",true);
                pipeline(gather,"photon_gather.comp",true);
                pipeline(temporal_pass,"photon_temporal.comp",true);
                pipeline(denoise,"photon_denoise.comp",true);
            }
        }
        void prepare(vk::CommandBuffer cmd)
        {
            const auto stages=vk::PipelineStageFlagBits::eComputeShader |
                              vk::PipelineStageFlagBits::eFragmentShader;
            const auto rw=vk::AccessFlagBits::eShaderRead |
                          vk::AccessFlagBits::eShaderWrite;
            auto transition=[&](gpu::image& x)
            {
                image_barrier(cmd,x,
                    prepared?vk::PipelineStageFlags(stages):
                             vk::PipelineStageFlags(vk::PipelineStageFlagBits::eTopOfPipe),
                    prepared?vk::AccessFlags(rw):vk::AccessFlags{},
                    vk::PipelineStageFlagBits::eComputeShader,rw,
                    prepared?vk::ImageLayout::eGeneral:
                             vk::ImageLayout::eUndefined,
                    vk::ImageLayout::eGeneral);
            };
            for(auto& x:history)transition(x);
            for(auto& x:positions)transition(x);
            transition(temporal);
            transition(scratch);
            prepared=true;
        }
        void atlas_transition(vk::CommandBuffer cmd,gpu::image& image,bool used)
        {
            const auto rw=vk::AccessFlagBits::eShaderRead |
                          vk::AccessFlagBits::eShaderWrite;
            image_barrier(cmd,image,
                used?vk::PipelineStageFlags(vk::PipelineStageFlagBits::eComputeShader):
                     vk::PipelineStageFlags(vk::PipelineStageFlagBits::eTopOfPipe),
                used?vk::AccessFlags(rw):vk::AccessFlags{},
                vk::PipelineStageFlagBits::eComputeShader,rw,
                used?vk::ImageLayout::eGeneral:
                     vk::ImageLayout::eUndefined,
                vk::ImageLayout::eGeneral);
        }
        photon_output transport(vk::CommandBuffer cmd)
        {
            if(!cfg.transport_only)
                throw std::logic_error("pm: record_transport requires transport_only configuration");
            if(!cmd)throw std::invalid_argument("pm: null transport command buffer");
            if(gi_samples>=cfg.gi_updates)
                return {sorted.handle(),counter.handle(),photon_capacity};
            camera_push push{};
            push.eye=pack(vec3{},float(frame));
            push.params={extent.width,extent.height,0u,cfg.bounces};
            push.config={cfg.photons,photon_capacity,gi_samples,node_count};
            push.settings={{cfg.photon_radius,0.f,0.f,0.f}};
            push.sphere=pack(world.glass.center,world.glass.radius);
            push.material={{world.glass.ior,0.f,float(cfg.atlas_tile),0.f}};
            prepare(cmd);
            const auto set=sets.set(0);
            clear.dispatch(cmd,(cells+127u)/128u,1,1,{set});
            compute_barrier(cmd);
            emit.dispatch(cmd,(cfg.photons+63u)/64u,1,1,{set},
                          &push,sizeof(push));
            compute_barrier(cmd);
            bin.dispatch(cmd,(photon_capacity+127u)/128u,1,1,{set},
                         &push,sizeof(push));
            compute_barrier(cmd);
            prefix.dispatch(cmd,1,1,1,{set});
            compute_barrier(cmd);
            scatter.dispatch(cmd,(photon_capacity+127u)/128u,1,1,{set},
                             &push,sizeof(push));
            compute_barrier(cmd);
            ++gi_samples;
            ++frame;
            return {sorted.handle(),counter.handle(),photon_capacity};
        }

        image_output render(vk::CommandBuffer cmd,const view& camera)
        {
            if(cfg.transport_only)
                throw std::logic_error("pm: fullscreen record disabled for transport-only mapper");
            if(!cmd)throw std::invalid_argument("pm: null command buffer");
            if(!(camera.vertical_fov>.05f && camera.vertical_fov<3.f))
                throw std::invalid_argument("pm: invalid camera FOV");
            const float half_tan=std::tan(camera.vertical_fov*.5f);
            const f4 eye=pack(camera.eye,float(frame));
            const f4 right=pack(camera.right);
            const f4 up=pack(camera.up,half_tan);
            bool moving=previous_valid &&
                (squared(vec3{camera.eye.x-last_eye[0],
                              camera.eye.y-last_eye[1],
                              camera.eye.z-last_eye[2]})>1e-10f ||
                 squared(vec3{camera.right.x-last_right[0],
                              camera.right.y-last_right[1],
                              camera.right.z-last_right[2]})>1e-10f ||
                 squared(vec3{camera.up.x-last_up[0],
                              camera.up.y-last_up[1],
                              camera.up.z-last_up[2]})>1e-10f ||
                 std::abs(half_tan-last_up[3])>1e-6f);
            stable_frames=(moving || !history_valid)?
                0u:std::min(stable_frames+1u,512u);
            const std::array<f4,3> previous{
                previous_valid?last_eye:eye,
                previous_valid?last_right:right,
                previous_valid?last_up:up
            };
            previous_camera.upload(previous.data(),sizeof(previous));
            current=std::uint32_t(frame&1u);
            const auto set=sets.set(current);
            const camera_push push{
                eye,right,up,
                {{extent.width,extent.height,stable_frames,cfg.bounces}},
                {{cfg.photons,photon_capacity,gi_samples,node_count}},
                {{cfg.photon_radius,history_valid?1.f:0.f,
                  cfg.screen_shadows?float(cfg.screen_spp):-float(cfg.edge_spp),
                  moving?1.f:0.f}},
                pack(world.glass.center,world.glass.radius),
                {{world.glass.ior,float(direct_samples),
                  float(cfg.atlas_tile),float(cfg.direct_spp)}}
            };
            prepare(cmd);
            const std::uint32_t gx=(extent.width+7u)/8u;
            const std::uint32_t gy=(extent.height+7u)/8u;
            const std::uint32_t ax=(atlas.width+7u)/8u;
            const std::uint32_t ay=(atlas.height+7u)/8u;
            if(gi_samples<cfg.gi_updates)
            {
                atlas_transition(cmd,indirect,indirect_prepared);
                clear.dispatch(cmd,(cells+127u)/128u,1,1,{set});
                compute_barrier(cmd);
                emit.dispatch(cmd,(cfg.photons+63u)/64u,1,1,{set},
                              &push,sizeof(push));
                compute_barrier(cmd);
                bin.dispatch(cmd,(photon_capacity+127u)/128u,1,1,
                             {set},&push,sizeof(push));
                compute_barrier(cmd);
                prefix.dispatch(cmd,1,1,1,{set});
                compute_barrier(cmd);
                scatter.dispatch(cmd,(photon_capacity+127u)/128u,1,1,
                                 {set},&push,sizeof(push));
                compute_barrier(cmd);
                gather_gi.dispatch(cmd,ax,ay,1,{set},&push,sizeof(push));
                compute_barrier(cmd);
                indirect_prepared=true;
                ++gi_samples;
            }
            // Full-screen direct rays bypass the surface atlas. Do not
            // spend thousands of shadow rays on a cache that is not read.
            if(direct_samples<cfg.direct_updates &&
               (!cfg.screen_shadows || cfg.debug_view>=3u))
            {
                atlas_transition(cmd,direct,direct_prepared);
                direct_pass.dispatch(cmd,ax,ay,1,{set},&push,sizeof(push));
                compute_barrier(cmd);
                direct_filter.dispatch(cmd,ax,ay,1,{set},&push,sizeof(push));
                compute_barrier(cmd);
                direct_prepared=true;
                ++direct_samples;
            }
            camera_push camera_shader=push;
            camera_shader.material[1]=float(cfg.debug_view);
            gather.dispatch(cmd,gx,gy,1,{set},&camera_shader,sizeof(camera_shader));
            compute_barrier(cmd);
            temporal_pass.dispatch(cmd,gx,gy,1,{set},&push,sizeof(push));
            compute_barrier(cmd);
            camera_push filter_shader=push;
            filter_shader.settings[3]=cfg.denoise_passes==1u?4.f:0.f;
            denoise.dispatch(cmd,gx,gy,1,{set},&filter_shader,sizeof(filter_shader));
            if(cfg.denoise_passes==2u)
            {
                compute_barrier(cmd);
                filter_shader.settings[3]=2.f;
                denoise.dispatch(cmd,gx,gy,1,{set},&filter_shader,sizeof(filter_shader));
            }
            gpu::image& display=cfg.denoise_passes==2u?temporal:scratch;
            image_barrier(cmd,display,
                 vk::PipelineStageFlagBits::eComputeShader,
                 vk::AccessFlagBits::eShaderWrite,
                 vk::PipelineStageFlagBits::eComputeShader |
                 vk::PipelineStageFlagBits::eFragmentShader,
                 vk::AccessFlagBits::eShaderRead,
                 vk::ImageLayout::eGeneral,vk::ImageLayout::eGeneral);
            image_barrier(cmd,positions[current],
                 vk::PipelineStageFlagBits::eComputeShader,
                 vk::AccessFlagBits::eShaderWrite,
                 vk::PipelineStageFlagBits::eFragmentShader,
                 vk::AccessFlagBits::eShaderRead,
                 vk::ImageLayout::eGeneral,vk::ImageLayout::eGeneral);
            have_output=true;
            history_valid=previous_valid=true;
            last_eye=eye;last_right=right;last_up=up;
            ++frame;
            return output();
        }
        image_output output() const
        {
            if(!have_output)return {};
            const gpu::image& display=cfg.denoise_passes==2u?temporal:scratch;
            return {display.handle(),display.view(),
                    extent,vk::ImageLayout::eGeneral};
        }
        image_output surface_positions() const
        {
            if(!have_output)return {};
            return {positions[current].handle(),positions[current].view(),
                    extent,vk::ImageLayout::eGeneral};
        }
        progress status() const
        {
            return {gi_samples,cfg.gi_updates,direct_samples,
                    cfg.direct_updates,frame};
        }
    };

    mapper::mapper()=default;
    mapper::~mapper()=default;
    void mapper::create(const scene& s,vk::Extent2D extent,
                        const configuration& cfg)
    {
        if(state_)throw std::logic_error("pm: mapper already created");
        auto next=std::make_unique<implementation>();
        next->create(s,extent,cfg);
        state_=std::move(next);
    }
    void mapper::destroy(){state_.reset();}
    void mapper::resize(vk::Extent2D extent)
    {
        if(!state_)throw std::logic_error("pm: mapper not created");
        const scene current=state_->world;
        const configuration cfg=state_->cfg;
        destroy();
        create(current,extent,cfg);
    }
    void mapper::set_scene(const scene& s)
    {
        if(!state_)throw std::logic_error("pm: mapper not created");
        const auto cfg=state_->cfg;
        const auto size=state_->extent;
        destroy();
        create(s,size,cfg);
    }
    void mapper::set_glass(const glass_sphere& sphere)
    {
        if(!state_)throw std::logic_error("pm: mapper not created");
        if(!std::isfinite(sphere.radius) || sphere.radius<0.f ||
           !(sphere.ior>1.f) || !std::isfinite(sphere.ior))
            throw std::invalid_argument("pm: invalid dielectric sphere");
        state_->world.glass=sphere;
        invalidate();
    }
    void mapper::set_emission(std::uint32_t index,vec3 emission)
    {
        if(!state_ || index>=state_->world.surfaces.size())
            throw std::out_of_range("pm: invalid emissive surface");
        const f4 value=pack(emission);
        state_->world.surfaces[index].emission=emission;
        state_->faces.upload(value.data(),sizeof(value),
                             std::size_t(index)*sizeof(gpu_face)+5u*sizeof(f4));
        invalidate();
    }
    void mapper::invalidate()
    {
        if(!state_)throw std::logic_error("pm: mapper not created");
        state_->gi_samples=state_->direct_samples=state_->stable_frames=0;
        state_->history_valid=state_->previous_valid=false;
    }
    image_output mapper::record(vk::CommandBuffer command,const view& v)
    {
        if(!state_)throw std::logic_error("pm: mapper not created");
        return state_->render(command,v);
    }
    photon_output mapper::record_transport(vk::CommandBuffer cmd)
    {
        if(!state_)throw std::logic_error("pm: mapper not created");
        return state_->transport(cmd);
    }
    photon_output mapper::photon_buffers() const
    {
        if(!state_)throw std::logic_error("pm: mapper not created");
        return {state_->sorted.handle(),state_->counter.handle(),
                state_->photon_capacity};
    }
    geometry_output mapper::geometry_buffers() const
    {
        if(!state_)throw std::logic_error("pm: mapper not created");
        return {state_->faces.handle(),state_->nodes.handle(),
                state_->triangles.handle(),state_->scene_info.handle(),
                state_->sorted.handle(),state_->offsets.handle()};
    }
    geometry_output mapper::refit_geometry(
        std::uint32_t frame,const std::vector<geometry_update>& updates)
    {
        if(!state_)throw std::logic_error("pm: mapper not created");
        return state_->refit_geometry(frame,updates);
    }
    image_output mapper::output() const
    {
        return state_?state_->output():image_output{};
    }
    image_output mapper::surface_positions() const
    {
        return state_?state_->surface_positions():image_output{};
    }
    progress mapper::status() const
    {
        return state_?state_->status():progress{};
    }
    const scene& mapper::current_scene() const
    {
        if(!state_)throw std::logic_error("pm: mapper not created");
        return state_->world;
    }
}
