#include "pcloud.hpp"
#include <render/renderer.hpp>
#include <render/pipeline.hpp>
#include <gpu/compute.hpp>
#include <gpu/descriptors.hpp>
#include <gpu/resource.hpp>
#include <bvh/stackless.hpp>
#include <spatial/bricks.hpp>
#include <jobs/pool.hpp>
#include <render/postprocess.hpp>
#include <memory>
#include <algorithm>
#include <atomic>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <cstdint>
#include <utility>
#include <iostream>
#include <numeric>
#include <functional>
#include <limits>
#include <optional>
namespace pc
{
    namespace
    {
        struct image_target {
            vk::Image image{};
            vk::ImageView view{};
        };

        struct frame {
            image_target hdr;
            image_target ldr;
            gpu::owned_buffer points;
            gpu::owned_buffer tile_lights;
            std::vector<std::uint32_t> tile_cpu;
            gpu::owned_image grid;
            vk::ImageView grid_view{};
            bool dirty=true;
            vk::DescriptorSet cloud{};
            vk::DescriptorSet gi{};
            vk::DescriptorSet lighting{};
            vk::DescriptorSet tone{};
            vk::DescriptorSet presentation{};
        };

        struct state {
            std::vector<frame> frames;
            vk::Sampler sampler{};
            gpu::owned_buffer lights;
            gpu::owned_buffer triangles;
            gpu::owned_buffer nodes;
            gpu::owned_buffer shadow_light_ranges;
            gpu::owned_buffer neighborhood_offsets;
            gpu::owned_buffer neighborhood_indices;
            std::shared_ptr<const world> snapshot;
            std::shared_ptr<const world> incoming;
            configuration shader_paths{};
            presentation_target output{};
            std::uint32_t light_count=0, point_count=0, vertex_count=0, node_count=0;
            std::uint32_t tile_columns=0, tile_rows=0;
            vk::util::math::vec3 grid_origin{};
            float grid_spacing=0.2f;
            vk::Extent3D grid_extent{1,1,1};
            std::unique_ptr<gpu::descriptors> cloud_sets;
            std::unique_ptr<gpu::descriptors> gi_sets;
            std::unique_ptr<gpu::descriptors> lighting_sets;
            std::unique_ptr<gpu::descriptors> tone_sets;
            std::unique_ptr<gpu::descriptors> present_sets;
            std::unique_ptr<::render::postprocess_targets> targets;
            std::unique_ptr<::render::presentation_targets> presentation;
            vk::PipelineLayout present_layout{};
            std::unique_ptr<gpu::compute> cloud_program;
            std::unique_ptr<gpu::compute> gi_program;
            std::unique_ptr<gpu::compute> lighting_program;
            std::unique_ptr<gpu::compute> tone_program;
            std::unique_ptr<::render::pipeline> present_program;
        };

        state& get()
        {
            static state data;
            return data;
        }

        std::atomic<float> exposure_value{};
        std::atomic<bool> draw_probes=false;
        constexpr float min_grid_spacing=0.2f;
        constexpr float bounce_reach=2.6f;
        constexpr std::uint64_t max_grid_probes=32768;
        constexpr std::uint32_t tile_size=16,light_slot_width=65;

        struct alignas(16) point_light {
            std::array<float,4> position_range{};
            std::array<float,4> color_intensity{};
        };
        static_assert(sizeof(point_light)==32);
        struct probe_point {
            std::array<float,4> position{};
            std::array<float,4> normal{};
            std::array<float,4> radiance{};
        };
        static_assert(sizeof(probe_point)==48);
        using shadow_node=bvh::stackless_node;
        struct cloud_push {
            std::uint32_t points,vertices,lights,nodes;
        };
        struct grid_push {
            std::uint32_t points,vertices,nodes,lights=0;
            std::array<float,4> origin_spacing{};
        };
        static_assert(sizeof(grid_push)==32);

        struct push {
            float exposure;
            std::array<float,3> padding{};
            std::array<float,4> eye_near{};
            std::array<float,4> forward_far{};
            std::array<float,4> right_tan{};
            std::array<float,4> up_tan{};
            std::array<float,4> light_count{};
            std::array<float,4> sky{};
        };
        static_assert(sizeof(push)==112);

        std::vector<vk::DescriptorSetLayoutBinding> bindings(
            std::uint32_t sampled,std::uint32_t storage,vk::ShaderStageFlags stages,
            std::uint32_t buffers=0)
        {
            std::vector<vk::DescriptorSetLayoutBinding> result(sampled+storage+buffers);
            for(std::uint32_t i=0;i<result.size();++i) {
                auto& binding=result[i];
                binding.binding=i;
                binding.descriptorCount=1;
                binding.stageFlags=stages;
                binding.descriptorType=i<sampled
                    ? vk::DescriptorType::eCombinedImageSampler
                    : i<sampled+storage ? vk::DescriptorType::eStorageImage
                                        : vk::DescriptorType::eStorageBuffer;
            }
            return result;
        }

        void create_grid(frame& current,vk::Extent3D extent)
        {
            if(current.grid_view)
            {
                vk::image_view::destroy(current.grid_view);
                current.grid_view=vk::ImageView{};
            }
            current.grid.reset();
            vk::ImageCreateInfo grid_info{};
            grid_info.imageType=vk::ImageType::e3D;
            grid_info.format=vk::Format::eR16G16B16A16Sfloat;
            grid_info.extent=extent;
            grid_info.mipLevels=1;
            grid_info.arrayLayers=1;
            grid_info.samples=vk::SampleCountFlagBits::e1;
            grid_info.tiling=vk::ImageTiling::eOptimal;
            grid_info.usage=vk::ImageUsageFlagBits::eStorage |
                            vk::ImageUsageFlagBits::eSampled;
            current.grid=gpu::create_image(grid_info,vk::MemoryPropertyFlagBits::eDeviceLocal);
            vk::ImageViewCreateInfo view{};
            view.image=current.grid.handle;
            view.viewType=vk::ImageViewType::e3D;
            view.format=grid_info.format;
            view.subresourceRange={vk::ImageAspectFlagBits::eColor,0,1,0,1};
            current.grid_view=vk::image_view::create(view);
            current.dirty=true;
        }

        vk::PipelineLayout pipeline_layout(vk::DescriptorSetLayout set)
        {
            vk::PipelineLayoutCreateInfo info{};
            info.setLayoutCount=1;
            info.pSetLayouts=&set;
            return vk::device::get()->handle.createPipelineLayout(info);
        }

        void compute(vk::CommandBuffer command,const gpu::compute& program,
                     vk::DescriptorSet set,const camera_frame& frame)
        {
            const auto extent=get().output.extent;
            const auto& view=frame;
            const float sy=std::sin(view.yaw),cy=std::cos(view.yaw);
            const float sp=std::sin(view.pitch),cp=std::cos(view.pitch);
            const float tan_half=std::tan(view.fov*0.5f);
            const float aspect=float(extent.width)/float(extent.height);
            const auto& position=frame.eye;
            const auto& grid=get();
            const push controls{
                exposure_value.load(std::memory_order_relaxed),{},
                {position.x,position.y,position.z,0.1f},
                {sy*cp,sp,-cy*cp,1000.0f},
                {cy,0,sy,tan_half*aspect},
                {-sy*sp,cp,cy*sp,tan_half},
                {float(get().light_count),draw_probes.load(std::memory_order_relaxed)?1.0f:0.0f,
                 float(get().vertex_count),float(get().node_count)},
                {grid.grid_origin.x,grid.grid_origin.y,grid.grid_origin.z,
                 grid.grid_spacing}
            };
            const auto width=extent.width;
            const auto height=extent.height;
            program.dispatch(command,(width+7)/8,(height+7)/8,1,
                             {set},&controls,sizeof(controls));
        }
    }

    void toggle_probes()
    {
        draw_probes.store(!draw_probes.load(std::memory_order_relaxed),
                          std::memory_order_relaxed);
    }

    void updated(std::uint32_t image)
    {
        get().frames.at(image).dirty=false;
    }

    void set_exposure(float exposure)
    {
        exposure_value.store(std::max(0.0f, exposure), std::memory_order_relaxed);
    }

    void create(const configuration& shaders,
                const std::vector<gbuffer>& gbuffers,
                const presentation_target& output)
    {
        auto& data = get();
        data.shader_paths=shaders;
        if(output.images.empty() || output.format==vk::Format::eUndefined ||
           !output.extent.width || !output.extent.height ||
           gbuffers.size()!=output.images.size())
            throw std::invalid_argument("pc presentation/G-buffer mismatch");
        data.output=output;
        const auto device = vk::device::get()->handle;
        try
        {
            vk::SamplerCreateInfo sampler{};
            sampler.magFilter = vk::Filter::eNearest;
            sampler.minFilter = vk::Filter::eNearest;
            sampler.mipmapMode = vk::SamplerMipmapMode::eNearest;
            sampler.addressModeU = vk::SamplerAddressMode::eClampToEdge;
            sampler.addressModeV = vk::SamplerAddressMode::eClampToEdge;
            sampler.addressModeW = vk::SamplerAddressMode::eClampToEdge;
            data.sampler = device.createSampler(sampler);

            const auto count=static_cast<std::uint32_t>(get().output.images.size());
            const auto extent=get().output.extent;
            data.tile_columns=(extent.width+tile_size-1u)/tile_size;
            data.tile_rows=(extent.height+tile_size-1u)/tile_size;
            data.targets=std::make_unique<::render::postprocess_targets>();
            data.targets->create(count,get().output.extent);
            data.presentation=std::make_unique<::render::presentation_targets>();
            data.presentation->create(get().output.images,
                get().output.format,get().output.extent);
            data.cloud_sets=std::make_unique<gpu::descriptors>();
            data.cloud_sets->create(bindings(0,0,vk::ShaderStageFlagBits::eCompute,4),count);
            data.gi_sets=std::make_unique<gpu::descriptors>();
            data.gi_sets->create(bindings(0,1,vk::ShaderStageFlagBits::eCompute,6),count);
            data.lighting_sets=std::make_unique<gpu::descriptors>();
            data.lighting_sets->create(bindings(5,1,vk::ShaderStageFlagBits::eCompute,5),count);
            data.tone_sets=std::make_unique<gpu::descriptors>();
            data.tone_sets->create(bindings(1,1,vk::ShaderStageFlagBits::eCompute),count);
            data.present_sets=std::make_unique<gpu::descriptors>();
            data.present_sets->create(bindings(1,0,vk::ShaderStageFlagBits::eFragment),count);
            data.present_layout=pipeline_layout(data.present_sets->layout());
            data.cloud_program=std::make_unique<gpu::compute>();
            data.cloud_program->create(data.shader_paths.cloud_shader,
                                       {data.cloud_sets->layout()},sizeof(cloud_push));
            data.gi_program=std::make_unique<gpu::compute>();
            data.gi_program->create(data.shader_paths.gi_shader,
                                    {data.gi_sets->layout()},sizeof(grid_push));
            data.lighting_program = std::make_unique<gpu::compute>();
            data.lighting_program->create(
                data.shader_paths.lighting_shader,
                {data.lighting_sets->layout()}, sizeof(push));
            data.tone_program = std::make_unique<gpu::compute>();
            data.tone_program->create(
                data.shader_paths.tonemap_shader,
                {data.tone_sets->layout()}, sizeof(push));

            data.present_program = std::make_unique<::render::pipeline>();
            ::render::raster_state raster{};
            raster.pass = data.presentation->pass();
            raster.layout = data.present_layout;
            raster.cull = vk::CullModeFlagBits::eNone;
            raster.depth_test = false;
            raster.depth_write = false;
            data.present_program->create(raster, {
                {vk::ShaderStageFlagBits::eVertex,
                 data.shader_paths.fullscreen_shader},
                {vk::ShaderStageFlagBits::eFragment,
                 data.shader_paths.presentation_shader}
            });

            const auto storage=vk::BufferUsageFlagBits::eStorageBuffer;
            const auto memory=vk::MemoryPropertyFlagBits::eHostVisible |
                              vk::MemoryPropertyFlagBits::eHostCoherent;
            data.lights=gpu::create_buffer(64*sizeof(point_light),storage,memory);
            data.triangles=gpu::create_buffer(3*sizeof(std::array<float,4>),storage,memory);
            data.nodes=gpu::create_buffer(sizeof(shadow_node),storage,memory);
            data.shadow_light_ranges=gpu::create_buffer(
                64*sizeof(bvh::light_tree_range),storage,memory);
            const std::array<bvh::light_tree_range,64> initial_ranges{};
            vk::buffer::write(data.shadow_light_ranges,initial_ranges.data(),
                              sizeof(initial_ranges));
            data.neighborhood_offsets=gpu::create_buffer(2*sizeof(std::uint32_t),storage,memory);
            data.neighborhood_indices=gpu::create_buffer(sizeof(std::uint32_t),storage,memory);

            data.frames.reserve(count);
            for(std::uint32_t i=0;i<count;++i)
            {
                data.frames.emplace_back();
                auto& current=data.frames.back();
                const auto attachments=data.targets->at(i);
                current.hdr={attachments.hdr,attachments.hdr_view};
                current.ldr={attachments.ldr,attachments.ldr_view};
                current.points=gpu::create_buffer(sizeof(probe_point),storage,memory);
                current.tile_cpu.resize(std::size_t(data.tile_columns)*
                    data.tile_rows*light_slot_width);
                current.tile_lights=gpu::create_buffer(
                    current.tile_cpu.size()*sizeof(std::uint32_t),storage,memory);

                create_grid(current,data.grid_extent);

                current.cloud=data.cloud_sets->set(i);
                current.gi=data.gi_sets->set(i);
                current.lighting=data.lighting_sets->set(i);
                current.tone=data.tone_sets->set(i);
                current.presentation=data.present_sets->set(i);

                const auto g = gbuffers.at(i);
                vk::DescriptorImageInfo grid_storage{};
                grid_storage.imageView=current.grid_view;
                grid_storage.imageLayout=vk::ImageLayout::eGeneral;
                data.gi_sets->write_image(i,0,grid_storage);
                data.cloud_sets->write_buffer(i,0,vk::buffer::descriptor(current.points));
                data.cloud_sets->write_buffer(i,1,vk::buffer::descriptor(data.triangles));
                data.cloud_sets->write_buffer(i,2,vk::buffer::descriptor(data.lights));
                data.cloud_sets->write_buffer(i,3,vk::buffer::descriptor(data.nodes));
                data.gi_sets->write_buffer(i,1,vk::buffer::descriptor(current.points));
                data.gi_sets->write_buffer(i,2,vk::buffer::descriptor(data.triangles));
                data.gi_sets->write_buffer(i,3,vk::buffer::descriptor(data.nodes));
                data.gi_sets->write_buffer(i,4,vk::buffer::descriptor(data.lights));
                data.gi_sets->write_buffer(i,5,vk::buffer::descriptor(data.neighborhood_offsets));
                data.gi_sets->write_buffer(i,6,vk::buffer::descriptor(data.neighborhood_indices));

                std::array<vk::DescriptorImageInfo,6> lighting{};
                for(std::size_t j=0;j<g.size();++j)
                {
                    lighting[j].sampler=data.sampler;
                    lighting[j].imageView=g[j];
                    lighting[j].imageLayout=j==3
                        ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
                        : vk::ImageLayout::eShaderReadOnlyOptimal;
                }
                lighting[4].sampler=data.sampler;
                lighting[4].imageView=current.grid_view;
                lighting[4].imageLayout=vk::ImageLayout::eGeneral;
                lighting[5].imageView=current.hdr.view;
                lighting[5].imageLayout=vk::ImageLayout::eGeneral;
                for(std::uint32_t j=0;j<lighting.size();++j)
                    data.lighting_sets->write_image(i,j,lighting[j]);
                data.lighting_sets->write_buffer(i,6,vk::buffer::descriptor(data.lights));
                data.lighting_sets->write_buffer(i,7,vk::buffer::descriptor(data.triangles));
                data.lighting_sets->write_buffer(i,8,vk::buffer::descriptor(data.nodes));
                data.lighting_sets->write_buffer(i,9,vk::buffer::descriptor(current.tile_lights));
                data.lighting_sets->write_buffer(i,10,vk::buffer::descriptor(data.shadow_light_ranges));
                std::array<vk::DescriptorImageInfo,2> tone{};
                tone[0].sampler = data.sampler;
                tone[0].imageView = current.hdr.view;
                tone[0].imageLayout = vk::ImageLayout::eGeneral;
                tone[1].imageView = current.ldr.view;
                tone[1].imageLayout = vk::ImageLayout::eGeneral;
                for(std::uint32_t j=0;j<tone.size();++j)
                    data.tone_sets->write_image(i,j,tone[j]);
                std::array<vk::DescriptorImageInfo,1> presented{};
                presented[0].sampler = data.sampler;
                presented[0].imageView = current.ldr.view;
                presented[0].imageLayout = vk::ImageLayout::eGeneral;
                data.present_sets->write_image(i,0,presented[0]);
            }
        }
        catch (...)
        {
            release();
            throw;
        }
    }

    // Rogue derives its point cloud from static floor/wall geometry.
    // Reuse the server's world-space collision surfaces without expanding
    // arbitrary imported mesh polygons into brute-force visibility rays.
    void set_scene(std::shared_ptr<const world> value)
    {
        get().incoming=std::move(value);
    }

    void synchronize_scene()
    {
        auto& data=get();
        const auto source=data.incoming;
        if(source==data.snapshot)return;

        using namespace vk::util::math;

        // Cover every server-authored light radius and its local bounce reach.
        // Preserve the full spatial coverage; reduce probe density if necessary
        // rather than silently cropping a light to a hardcoded box.
        vec3 origin{};
        float spacing=min_grid_spacing;
        vk::Extent3D extent{1,1,1};
        if(source && !source->lights.empty())
        {
            const float inf=std::numeric_limits<float>::infinity();
            vec3 minimum{inf,inf,inf},maximum{-inf,-inf,-inf};
            bool any=false;
            for(const auto& light:source->lights)
            {
                if(!std::isfinite(light.position.x) ||
                   !std::isfinite(light.position.y) ||
                   !std::isfinite(light.position.z) ||
                   !std::isfinite(light.radius) || light.radius<=0.0f ||
                   !std::isfinite(light.strength))
                    throw std::runtime_error("Invalid streamed light bounds");
                if(light.strength<=0.0f)continue;
                const float reach=light.radius+bounce_reach;
                const vec3 low=light.position-vec3{reach,reach,reach};
                const vec3 high=light.position+vec3{reach,reach,reach};
                minimum={std::min(minimum.x,low.x),
                         std::min(minimum.y,low.y),
                         std::min(minimum.z,low.z)};
                maximum={std::max(maximum.x,high.x),
                         std::max(maximum.y,high.y),
                         std::max(maximum.z,high.z)};
                any=true;
            }
            if(any)
            {
                const vec3 dimensions=maximum-minimum;
                const auto max_dim=vk::physical_device::get()->handle
                    .getProperties().limits.maxImageDimension3D;
                const double dx=dimensions.x,dy=dimensions.y,dz=dimensions.z;
                double step=std::max({
                    double(min_grid_spacing),
                    std::max({dx,dy,dz})/std::max(2u,max_dim-1u),
                    std::cbrt((dx*dy*dz)/double(max_grid_probes))
                });
                for(;;)
                {
                    const auto nx=std::uint32_t(std::ceil(dx/step))+1u;
                    const auto ny=std::uint32_t(std::ceil(dy/step))+1u;
                    const auto nz=std::uint32_t(std::ceil(dz/step))+1u;
                    if(nx<=max_dim && ny<=max_dim && nz<=max_dim &&
                       std::uint64_t(nx)*ny*nz<=max_grid_probes)
                    {
                        extent=vk::Extent3D{nx,ny,nz};
                        break;
                    }
                    step*=1.1;
                }
                origin=minimum;
                spacing=static_cast<float>(step);
            }
        }
        const vec3 limit{
            origin.x+(extent.width-1)*spacing,
            origin.y+(extent.height-1)*spacing,
            origin.z+(extent.depth-1)*spacing
        };
        std::vector<point_light> lights;
        std::vector<probe_point> points;
        std::vector<std::array<float,4>> triangles;
        if(source)
        {
            if(source->lights.size()>64)
                throw std::runtime_error("Scene exceeds 64 point lights");
            for(const auto& light:source->lights)
                lights.push_back({
                    {light.position.x,light.position.y,light.position.z,light.radius},
                    {light.color.x,light.color.y,light.color.z,light.strength}
                });

            // Rogue's probe cloud is built from static room surfaces, not
            // every polygon of imported character/prop meshes. The server
            // already streams those static surfaces as collision boxes.
            const auto face=[&](vec3 base,vec3 u,vec3 v,vec3 normal)
            {
                // Samples use the original 0.3-unit surface spacing.
                // Restrict iteration to the probe volume, even for a huge floor.
                const float lu=length(u),lv=length(v);
                if(lu<=0.0f || lv<=0.0f)return;
                const vec3 U=u/lu,V=v/lv;
                const int nu=std::max(1,int(std::ceil(lu/0.3f)));
                const int nv=std::max(1,int(std::ceil(lv/0.3f)));
                const float du=lu/float(nu),dv=lv/float(nv);
                const auto interval=[&](vec3 axis,float step,int count)
                {
                    float lo=dot(origin-base,axis);
                    float hi=dot(limit-base,axis);
                    if(lo>hi)std::swap(lo,hi);
                    const int begin=std::clamp(int(std::ceil(lo/step-0.5f)),0,count);
                    const int finish=std::clamp(int(std::ceil(hi/step+0.5f)),0,count);
                    return std::pair{begin,finish};
                };
                const auto [u0,u1]=interval(U,du,nu);
                const auto [v0,v1]=interval(V,dv,nv);
                for(int i=u0;i<u1;++i)
                for(int j=v0;j<v1;++j)
                {
                    const vec3 p=base+U*((float(i)+0.5f)*du)
                                     +V*((float(j)+0.5f)*dv);
                    if(p.x<origin.x || p.y<origin.y || p.z<origin.z ||
                       p.x>limit.x || p.y>limit.y || p.z>limit.z)continue;
                    // Do not generate surface samples in dark gaps between lights.
                    bool reached=false;
                    for(const auto& light:source->lights)
                    {
                        const float radius=light.radius+bounce_reach;
                        if(light.strength>0.0f &&
                           length_squared(p-light.position)<=radius*radius)
                        {
                            reached=true;
                            break;
                        }
                    }
                    if(!reached)continue;
                    points.push_back({
                        {p.x,p.y,p.z,0.0f},
                        {normal.x,normal.y,normal.z,0.0f},{}
                    });
                }
            };

            for(const auto& box:source->collision_boxes)
            {
                // Keep all casters that can intersect a light's radius,
                // including those just outside the probe volume.
                const bool inside=
                    box.maximum.x>=origin.x-bounce_reach &&
                    box.maximum.y>=origin.y-bounce_reach &&
                    box.maximum.z>=origin.z-bounce_reach &&
                    box.minimum.x<=limit.x+bounce_reach &&
                    box.minimum.y<=limit.y+bounce_reach &&
                    box.minimum.z<=limit.z+bounce_reach;
                bool near_light=false;
                for(const auto& light:source->lights)
                {
                    const vec3 c{
                        std::clamp(light.position.x,box.minimum.x,box.maximum.x),
                        std::clamp(light.position.y,box.minimum.y,box.maximum.y),
                        std::clamp(light.position.z,box.minimum.z,box.maximum.z)
                    };
                    if(length_squared(c-light.position)<=light.radius*light.radius)
                    {
                        near_light=true;
                        break;
                    }
                }
                if(!inside && !near_light)continue;
                const auto a=box.minimum,b=box.maximum;
                const float dx=b.x-a.x,dy=b.y-a.y,dz=b.z-a.z;
                face({b.x,a.y,a.z},{0,dy,0},{0,0,dz},{1,0,0});
                face({a.x,a.y,b.z},{0,dy,0},{0,0,-dz},{-1,0,0});
                face({a.x,b.y,b.z},{dx,0,0},{0,0,-dz},{0,1,0});
                face({a.x,a.y,a.z},{dx,0,0},{0,0,dz},{0,-1,0});
                face({a.x,a.y,b.z},{dx,0,0},{0,dy,0},{0,0,1});
                face({b.x,a.y,a.z},{-dx,0,0},{0,dy,0},{0,0,-1});
            }
            // Host provides world-space triangles including animation skinning.
            for(const auto& triangle:source->triangles)
            {
                const vec3 a=triangle.a,b=triangle.b,c=triangle.c;
                if(length_squared(cross(b-a,c-a))<1e-12f)continue;
                if(!relevant_triangle(triangle,source->lights))continue;
                for(const auto& p:{a,b,c})
                    triangles.push_back({p.x,p.y,p.z,0.0f});
            }
        }
        // Build independent CPU acceleration structures concurrently.
        // Only immutable point coordinates cross to the job; GPU resources
        // and descriptor writes remain owned by the render thread.
        std::vector<vec3> surface_positions;
        surface_positions.reserve(points.size());
        for(const auto& point:points)
            surface_positions.push_back({
                point.position[0],point.position[1],point.position[2]
            });
        jobs::pool preparation;
        preparation.start(1);
        std::optional<spatial::bricked_neighborhood> indexed;
        if(!preparation.submit(
            [positions=std::move(surface_positions),origin,spacing,extent,
             &indexed]() mutable ->jobs::completion
            {
                auto result=spatial::build_bricked_neighborhood(
                    positions,origin,spacing,
                    {extent.width,extent.height,extent.depth},
                    bounce_reach+0.0001f);
                return [&indexed,result=std::move(result)]() mutable
                {
                    indexed=std::move(result);
                };
            }))
            throw std::runtime_error("Failed to schedule lighting preparation");
        auto nodes=bvh::build_stackless(triangles);
        // The original tree remains at the beginning of both buffers; GI
        // and probe-cloud queries continue using its original counts.
        const auto shared_nodes=static_cast<std::uint32_t>(nodes.size());
        const auto shared_vertices=static_cast<std::uint32_t>(triangles.size());
        std::vector<bvh::local_light> local_lights;
        local_lights.reserve(lights.size());
        for(const auto& light:lights)
            local_lights.push_back({
                {light.position_range[0],light.position_range[1],
                 light.position_range[2]},light.position_range[3]
            });
        const auto light_ranges=bvh::append_light_local_trees(
            triangles,nodes,local_lights);
        // Both the shared tree and local subtrees now reference immutable
        // triangles. Encode (origin, edge1, edge2) for all shadow queries.
        bvh::store_triangle_edges(triangles);
        preparation.wait_idle();
        preparation.commit(1);
        auto neighborhoods=std::move(indexed.value());
        std::size_t accelerated_lights=0,unshadowed_lights=0;
        for(const auto& range:light_ranges)
        {
            if(range.first>0 && range.count>0)++accelerated_lights;
            if(range.count==0)++unshadowed_lights;
        }
        std::clog << "clt: Rogue lighting geometry: "
                  << points.size() << " probes, "
                  << shared_vertices/3 << " mesh triangles, "
                  << shared_nodes << " shared shadow nodes, "
                  << nodes.size()-shared_nodes << " local shadow nodes, "
                  << accelerated_lights << "/" << light_ranges.size()
                  << " accelerated lights, "
                  << unshadowed_lights << " lights without casters\n";

        const bool new_shape=
            extent.width!=data.grid_extent.width ||
            extent.height!=data.grid_extent.height ||
            extent.depth!=data.grid_extent.depth;
        data.grid_origin=origin;
        data.grid_spacing=spacing;
        data.grid_extent=extent;
        if(new_shape)
        {
            // synchronize_scene() is called after prior GPU frames are idle.
            for(std::uint32_t i=0;i<data.frames.size();++i)
            {
                auto& frame=data.frames[i];
                create_grid(frame,extent);
                vk::DescriptorImageInfo gi{};
                gi.imageView=frame.grid_view;
                gi.imageLayout=vk::ImageLayout::eGeneral;
                data.gi_sets->write_image(i,0,gi);
                gi.sampler=data.sampler;
                data.lighting_sets->write_image(i,4,gi);
            }
        }
        std::clog << "clt: light-driven GI volume: "
                  << extent.width << "x" << extent.height << "x"
                  << extent.depth << " probes, spacing=" << spacing
                  << ", origin=(" << origin.x << "," << origin.y
                  << "," << origin.z << ")\n";

        const auto storage=vk::BufferUsageFlagBits::eStorageBuffer;
        const auto memory=vk::MemoryPropertyFlagBits::eHostVisible |
                          vk::MemoryPropertyFlagBits::eHostCoherent;
        auto new_triangles=gpu::create_buffer(
            std::max<std::size_t>(3*sizeof(std::array<float,4>),
                                  triangles.size()*sizeof(std::array<float,4>)),
            storage,memory);
        if(!triangles.empty())
            vk::buffer::write(new_triangles,triangles.data(),
                              triangles.size()*sizeof(std::array<float,4>));
        auto new_nodes=gpu::create_buffer(
            std::max<std::size_t>(sizeof(shadow_node),
                                  nodes.size()*sizeof(shadow_node)),
            storage,memory);
        if(!nodes.empty())
            vk::buffer::write(new_nodes,nodes.data(),nodes.size()*sizeof(shadow_node));
        if(!lights.empty())
            vk::buffer::write(data.lights,lights.data(),
                              lights.size()*sizeof(point_light));

        data.triangles=std::move(new_triangles);
        data.nodes=std::move(new_nodes);
        if(!light_ranges.empty())
            vk::buffer::write(data.shadow_light_ranges,
                light_ranges.data(),light_ranges.size()*sizeof(bvh::light_tree_range));
        // Ordered candidate lists keep the original sum order for each cell.
        const auto offset_bytes=neighborhoods.offsets.size()*sizeof(std::uint32_t);
        const auto candidate_bytes=neighborhoods.indices.size()*sizeof(std::uint32_t);
        auto new_offsets=gpu::create_buffer(
            std::max<std::size_t>(2*sizeof(std::uint32_t),offset_bytes),storage,memory);
        auto new_candidates=gpu::create_buffer(
            std::max<std::size_t>(sizeof(std::uint32_t),candidate_bytes),storage,memory);
        vk::buffer::write(new_offsets,neighborhoods.offsets.data(),offset_bytes);
        if(candidate_bytes)
            vk::buffer::write(new_candidates,neighborhoods.indices.data(),candidate_bytes);
        data.neighborhood_offsets=std::move(new_offsets);
        data.neighborhood_indices=std::move(new_candidates);
        data.node_count=shared_nodes;
        data.point_count=static_cast<std::uint32_t>(points.size());
        data.vertex_count=shared_vertices;
        data.light_count=static_cast<std::uint32_t>(lights.size());
        for(std::uint32_t i=0;i<data.frames.size();++i)
        {
            auto& frame=data.frames[i];
            frame.points=gpu::create_buffer(
                std::max<std::size_t>(sizeof(probe_point),points.size()*sizeof(probe_point)),
                storage,memory);
            if(!points.empty())
                vk::buffer::write(frame.points,points.data(),points.size()*sizeof(probe_point));
            data.cloud_sets->write_buffer(i,0,vk::buffer::descriptor(frame.points));
            data.cloud_sets->write_buffer(i,1,vk::buffer::descriptor(data.triangles));
            data.gi_sets->write_buffer(i,1,vk::buffer::descriptor(frame.points));
            data.gi_sets->write_buffer(i,2,vk::buffer::descriptor(data.triangles));
            data.lighting_sets->write_buffer(i,7,vk::buffer::descriptor(data.triangles));
            data.cloud_sets->write_buffer(i,3,vk::buffer::descriptor(data.nodes));
            data.gi_sets->write_buffer(i,3,vk::buffer::descriptor(data.nodes));
            data.gi_sets->write_buffer(i,5,vk::buffer::descriptor(data.neighborhood_offsets));
            data.gi_sets->write_buffer(i,6,vk::buffer::descriptor(data.neighborhood_indices));
            data.lighting_sets->write_buffer(i,8,vk::buffer::descriptor(data.nodes));
        }
        data.snapshot=source;
        for(auto& frame:data.frames)frame.dirty=true;
    }

    void release()
    {
        auto& data = get();
        const auto device = vk::device::get()->handle;
        for(auto& frame:data.frames)
        {
            if(frame.grid_view)vk::image_view::destroy(frame.grid_view);
            frame.grid_view=vk::ImageView{};
        }
        data.frames.clear();
        data.cloud_program.reset();
        data.gi_program.reset();
        data.lighting_program.reset();
        data.tone_program.reset();
        data.present_program.reset();
        data.presentation.reset();
        if (data.present_layout) device.destroyPipelineLayout(data.present_layout);
        data.cloud_sets.reset();
        data.gi_sets.reset();
        data.lighting_sets.reset();
        data.tone_sets.reset();
        data.present_sets.reset();
        data.targets.reset();
        data.lights.reset();
        data.triangles.reset();
        data.nodes.reset();
        data.shadow_light_ranges.reset();
        data.neighborhood_offsets.reset();
        data.neighborhood_indices.reset();
        if (data.sampler) device.destroySampler(data.sampler);

        data = {};
    }

    targets images(std::uint32_t image)
    {
        const auto& frame = get().frames.at(image);
        return {frame.grid.handle,frame.points.handle,frame.hdr.image,frame.ldr.image};
    }

    void cloud(vk::CommandBuffer command,std::uint32_t image)
    {
        if(!get().frames.at(image).dirty || !get().point_count)return;
        const cloud_push push{get().point_count,get().vertex_count,get().light_count,get().node_count};
        get().cloud_program->dispatch(command,(get().point_count+63)/64,1,1,
                                      {get().frames.at(image).cloud},&push,sizeof(push));
    }

    void gi(vk::CommandBuffer command,std::uint32_t image)
    {
        if(!get().frames.at(image).dirty)return;
        const grid_push push{
            get().point_count,get().vertex_count,get().node_count,get().light_count,
            {get().grid_origin.x,get().grid_origin.y,get().grid_origin.z,
             get().grid_spacing}
        };
        get().gi_program->dispatch(command,
            (get().grid_extent.width+3)/4,
            (get().grid_extent.height+3)/4,
            (get().grid_extent.depth+3)/4,
            {get().frames.at(image).gi},&push,sizeof(push));
    }

    void lighting(vk::CommandBuffer command, std::uint32_t image,
                  const camera_frame& frame)
    {
        auto& data=get();
        auto& current=data.frames.at(image);
        // Conservatively rasterize each light's projected influence sphere into
        // screen tiles. Rays that cannot intersect the sphere cannot contribute.
        // The remaining light IDs retain the original server-defined order.
        std::fill(current.tile_cpu.begin(),current.tile_cpu.end(),0u);
        if(data.snapshot)
        {
            using namespace vk::util::math;
            const auto extent=get().output.extent;
            const float sy=std::sin(frame.yaw),cy=std::cos(frame.yaw);
            const float sp=std::sin(frame.pitch),cp=std::cos(frame.pitch);
            const vec3 forward{sy*cp,sp,-cy*cp};
            const vec3 right{cy,0,sy};
            const vec3 up{-sy*sp,cp,cy*sp};
            const float tan_v=std::tan(frame.fov*0.5f);
            const float tan_h=tan_v*float(extent.width)/float(extent.height);
            const auto projected=[](float value,float z,float radius,float tangent)
            {
                const float a=(value-radius)/(z-radius);
                const float b=(value-radius)/(z+radius);
                const float c=(value+radius)/(z-radius);
                const float d=(value+radius)/(z+radius);
                return std::pair{std::min(a,b)/tangent,std::max(c,d)/tangent};
            };
            for(std::uint32_t i=0;i<data.snapshot->lights.size();++i)
            {
                const auto& light=data.snapshot->lights[i];
                if(light.radius<=0.f || light.strength==0.f)continue;
                const vec3 delta=light.position-frame.eye;
                const float z=dot(delta,forward),r=light.radius;
                std::uint32_t x0=0,x1=data.tile_columns;
                std::uint32_t y0=0,y1=data.tile_rows;
                if(z>r && tan_v>0.f && tan_h>0.f)
                {
                    const auto [left,right_edge]=projected(dot(delta,right),z,r,tan_h);
                    const auto [bottom,top]=projected(dot(delta,up),z,r,tan_v);
                    const auto lower=[](float ndc,std::uint32_t size,
                                        std::uint32_t count)
                    {
                        return std::uint32_t(std::clamp(
                            int(std::floor((ndc*0.5f+0.5f)*float(size)/tile_size))-1,
                            0,int(count)));
                    };
                    const auto upper=[](float ndc,std::uint32_t size,
                                        std::uint32_t count)
                    {
                        return std::uint32_t(std::clamp(
                            int(std::ceil((ndc*0.5f+0.5f)*float(size)/tile_size))+1,
                            0,int(count)));
                    };
                    x0=lower(left,extent.width,data.tile_columns);
                    x1=upper(right_edge,extent.width,data.tile_columns);
                    y0=lower(-top,extent.height,data.tile_rows);
                    y1=upper(-bottom,extent.height,data.tile_rows);
                }
                for(std::uint32_t y=y0;y<y1;++y)
                for(std::uint32_t x=x0;x<x1;++x)
                {
                    const auto base=(std::size_t(y)*data.tile_columns+x)*light_slot_width;
                    auto& count=current.tile_cpu[base];
                    if(count<light_slot_width-1u)
                        current.tile_cpu[base+1u+count++]=i;
                }
            }
        }
        vk::buffer::write(current.tile_lights,current.tile_cpu.data(),
                          current.tile_cpu.size()*sizeof(std::uint32_t));
        compute(command,*data.lighting_program,current.lighting,frame);
    }

    void tonemap(vk::CommandBuffer command, std::uint32_t image,
                 const camera_frame& frame)
    {
        const auto& current = get().frames.at(image);
        compute(command,*get().tone_program,current.tone,frame);
    }

    void present(vk::CommandBuffer command, std::uint32_t image,
                 const overlay_callback& overlay)
    {
        const auto extent = get().output.extent;
        vk::ClearValue clear{};
        clear.color = vk::ClearColorValue(std::array<float, 4>{0, 0, 0, 1});
        vk::RenderPassBeginInfo info{};
        info.renderPass = get().presentation->pass();
        info.framebuffer = get().presentation->framebuffer(image);
        info.renderArea = vk::Rect2D{{0, 0}, extent};
        info.clearValueCount = 1;
        info.pClearValues = &clear;

        ::render::renderer graphics(command);
        graphics.begin_pass(info);
        graphics.viewport(
            vk::Viewport{0, 0, float(extent.width), float(extent.height), 0, 1},
            vk::Rect2D{{0, 0}, extent});
        ::render::mesh triangle{};
        triangle.vertex_count = 3;
        ::render::material surface{};
        surface.pipeline = get().present_program->handle();
        surface.layout = get().present_layout;
        surface.sets = {get().frames.at(image).presentation};
        graphics.draw(triangle, surface);
        if(overlay)overlay(command,extent);
        graphics.end_pass();
    }
}
