#include <pt/pt.hpp>
#include <gpu/buffer.hpp>
#include <gpu/image.hpp>
#include <gpu/descriptors.hpp>
#include <gpu/compute.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace pt
{
    namespace
    {
        using v3=vk::util::math::vec3;
        using v4=vk::util::math::vec4;
        struct alignas(16) gpu_face
        {
            v4 origin{},u{},v{},normal{},albedo{},emission{},metric{};
        };
        static_assert(sizeof(gpu_face)==112);
        struct alignas(16) camera_push
        {
            v4 eye{},right{},up{};
            std::array<std::uint32_t,4> params{};
            v4 settings{};
            std::array<std::uint32_t,4> scene{};
        };
        static_assert(sizeof(camera_push)==96);
        v4 pack(v3 a,float w=0.f){return {a.x,a.y,a.z,w};}
        std::vector<gpu_face> prepare(const scene& world)
        {
            if(world.surfaces.empty() || world.surfaces.size()>1024u ||
               world.emitter>=world.surfaces.size())
                throw std::invalid_argument("pt: expected 1..1024 surfaces and a valid emitter");
            std::vector<gpu_face> result;
            result.reserve(world.surfaces.size());
            for(const surface& f:world.surfaces)
            {
                const float a=vk::util::math::dot(f.u,f.u);
                const float b=vk::util::math::dot(f.u,f.v);
                const float c=vk::util::math::dot(f.v,f.v);
                const float determinant=a*c-b*b;
                const float area=vk::util::math::length(
                    vk::util::math::cross(f.u,f.v))*(f.triangle?.5f:1.f);
                const float normal2=vk::util::math::dot(f.normal,f.normal);
                if(!(determinant>1e-8f) || !(area>1e-6f) ||
                   !(normal2>1e-8f) || !std::isfinite(area))
                    throw std::invalid_argument("pt: degenerate surface");
                result.push_back({pack(f.origin),pack(f.u),pack(f.v),
                    pack(f.normal*(1.f/std::sqrt(normal2)),f.triangle?1.f:0.f),
                    pack(f.albedo),pack(f.emission),
                    {c/determinant,-b/determinant,a/determinant,area}});
            }
            return result;
        }
        void validate(const configuration& cfg,vk::Extent2D extent)
        {
            if(!extent.width || !extent.height || cfg.bounces<1u ||
               cfg.bounces>8u || (cfg.light_spp!=1u && cfg.light_spp!=2u &&
               cfg.light_spp!=4u) || cfg.denoise_radius>2u ||
               cfg.shader_directory.empty())
                throw std::invalid_argument("pt: invalid render configuration");
        }
        void compute_barrier(vk::CommandBuffer cmd)
        {
            vk::MemoryBarrier b{};
            b.srcAccessMask=vk::AccessFlagBits::eShaderWrite;
            b.dstAccessMask=vk::AccessFlagBits::eShaderRead |
                            vk::AccessFlagBits::eShaderWrite;
            cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                vk::PipelineStageFlagBits::eComputeShader,
                                {},{b},{},{});
        }
        void image_barrier(vk::CommandBuffer cmd,gpu::image& resource,
                           vk::PipelineStageFlags from,vk::AccessFlags reads,
                           vk::PipelineStageFlags to,vk::AccessFlags writes,
                           vk::ImageLayout old_layout,vk::ImageLayout new_layout)
        {
            vk::ImageMemoryBarrier b{};
            b.srcAccessMask=reads;
            b.dstAccessMask=writes;
            b.oldLayout=old_layout;
            b.newLayout=new_layout;
            b.srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
            b.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
            b.image=resource.handle();
            b.subresourceRange.aspectMask=vk::ImageAspectFlagBits::eColor;
            b.subresourceRange.levelCount=1;
            b.subresourceRange.layerCount=1;
            cmd.pipelineBarrier(from,to,{},{},{},{b});
        }
    }

    struct tracer::implementation
    {
        pt::scene world;
        configuration cfg;
        vk::Extent2D extent{};
        gpu::buffer faces,previous_camera;
        std::array<gpu::image,2> history,positions;
        gpu::image display,temporal;
        gpu::descriptors descriptors;
        gpu::compute trace,accumulate,filter;
        std::uint64_t frame=0;
        std::uint32_t current=0,stable=0;
        bool initialized=false,valid_history=false,moved=false,have_output=false;
        v4 last_eye{},last_right{},last_up{};

        void create(pt::scene source,vk::Extent2D size,configuration options)
        {
            validate(options,size);
            auto gpu_faces=prepare(source);
            cfg=std::move(options);
            world=std::move(source);
            extent=size;
            const auto host=vk::MemoryPropertyFlagBits::eHostVisible |
                            vk::MemoryPropertyFlagBits::eHostCoherent;
            const auto storage=vk::BufferUsageFlagBits::eStorageBuffer;
            faces.create(sizeof(gpu_face)*gpu_faces.size(),storage,host);
            faces.upload(gpu_faces.data(),sizeof(gpu_face)*gpu_faces.size());
            previous_camera.create(sizeof(v4)*3u,storage,host);

            vk::ImageCreateInfo image{};
            image.imageType=vk::ImageType::e2D;
            image.format=vk::Format::eR32G32B32A32Sfloat;
            image.extent.width=size.width;
            image.extent.height=size.height;
            image.extent.depth=1;
            image.mipLevels=image.arrayLayers=1;
            image.samples=vk::SampleCountFlagBits::e1;
            image.tiling=vk::ImageTiling::eOptimal;
            image.usage=vk::ImageUsageFlagBits::eStorage;
            image.sharingMode=vk::SharingMode::eExclusive;
            vk::ImageViewCreateInfo view{};
            view.viewType=vk::ImageViewType::e2D;
            view.format=image.format;
            view.subresourceRange.aspectMask=vk::ImageAspectFlagBits::eColor;
            view.subresourceRange.levelCount=view.subresourceRange.layerCount=1;
            const auto allocate=[&](gpu::image& resource)
            {
                resource.create(image,vk::MemoryPropertyFlagBits::eDeviceLocal);
                resource.make_view(view);
            };
            for(auto& resource:history)allocate(resource);
            image.format=vk::Format::eR16G16B16A16Sfloat;
            view.format=image.format;
            for(auto& resource:positions)allocate(resource);
            allocate(display);
            allocate(temporal);

            std::vector<vk::DescriptorSetLayoutBinding> bindings;
            for(std::uint32_t i=0;i<8u;++i)
            {
                vk::DescriptorSetLayoutBinding b{};
                b.binding=i;
                b.descriptorType=(i==1u || i==7u)?
                    vk::DescriptorType::eStorageBuffer:
                    vk::DescriptorType::eStorageImage;
                b.descriptorCount=1;
                b.stageFlags=vk::ShaderStageFlagBits::eCompute;
                bindings.push_back(b);
            }
            descriptors.create(std::move(bindings),2);
            const auto assign_image=[&](std::uint32_t set,std::uint32_t binding,
                                        const gpu::image& resource)
            {
                vk::DescriptorImageInfo info{};
                info.imageView=resource.view();
                info.imageLayout=vk::ImageLayout::eGeneral;
                descriptors.write_image(set,binding,info);
            };
            for(std::uint32_t set=0;set<2u;++set)
            {
                assign_image(set,0,history[set]);
                descriptors.write_buffer(set,1,faces.descriptor());
                assign_image(set,2,positions[set]);
                assign_image(set,3,positions[set^1u]);
                assign_image(set,4,display);
                assign_image(set,5,history[set^1u]);
                assign_image(set,6,temporal);
                descriptors.write_buffer(set,7,previous_camera.descriptor());
            }
            const std::string prefix=cfg.shader_directory+"/pt_";
            const auto layout=descriptors.layout();
            trace.create(prefix+"trace.comp.spv",{layout},sizeof(camera_push));
            accumulate.create(prefix+"temporal.comp.spv",{layout},sizeof(camera_push));
            filter.create(prefix+"denoise.comp.spv",{layout},sizeof(camera_push));
        }
        void record(vk::CommandBuffer cmd,const view& camera)
        {
            if(!cmd || !std::isfinite(camera.vertical_fov) ||
               !(camera.vertical_fov>0.01f && camera.vertical_fov<3.1f))
                throw std::invalid_argument("pt: invalid command or camera FOV");
            const float fov=std::tan(camera.vertical_fov*.5f);
            const v4 eye=pack(camera.eye,float(frame));
            const v4 right=pack(camera.right);
            const v4 up=pack(camera.up,fov);
            const auto diff=[](v4 a,v4 b)
            {
                return std::fabs(a.x-b.x)+std::fabs(a.y-b.y)+
                       std::fabs(a.z-b.z);
            };
            moved=valid_history &&
                (diff(eye,last_eye)>1e-6f ||
                 diff(right,last_right)>1e-6f ||
                 diff(up,last_up)>1e-6f ||
                 std::fabs(up.w-last_up.w)>1e-6f);
            stable=(!valid_history || moved)?0u:
                    std::min(stable+1u,1000000u);
            const auto rw=vk::AccessFlagBits::eShaderRead |
                          vk::AccessFlagBits::eShaderWrite;
            const auto prior=vk::PipelineStageFlagBits::eComputeShader |
                             vk::PipelineStageFlagBits::eFragmentShader;
            for(gpu::image* image:{&history[0],&history[1],
                                   &positions[0],&positions[1],
                                   &display,&temporal})
                image_barrier(cmd,*image,
                    initialized?vk::PipelineStageFlags(prior):
                                vk::PipelineStageFlags(vk::PipelineStageFlagBits::eTopOfPipe),
                    initialized?vk::AccessFlags(rw):vk::AccessFlags{},
                    vk::PipelineStageFlagBits::eComputeShader,rw,
                    initialized?vk::ImageLayout::eGeneral:vk::ImageLayout::eUndefined,
                    vk::ImageLayout::eGeneral);
            initialized=true;
            current=std::uint32_t(frame&1u);
            const std::array<v4,3> previous{
                valid_history?last_eye:eye,
                valid_history?last_right:right,
                valid_history?last_up:up
            };
            previous_camera.upload(previous.data(),sizeof(previous));
            const camera_push push{
                eye,right,up,
                {extent.width,extent.height,stable,cfg.bounces},
                {valid_history&&cfg.temporal?1.f:0.f,
                 moved?1.f:0.f,float(cfg.light_spp),
                 float(cfg.denoise_radius)},
                {std::uint32_t(world.surfaces.size()),world.emitter,0u,0u}
            };
            const auto set=descriptors.set(current);
            const auto group_x=(extent.width+7u)/8u;
            const auto group_y=(extent.height+7u)/8u;
            trace.dispatch(cmd,group_x,group_y,1,{set},&push,sizeof(push));
            compute_barrier(cmd);
            accumulate.dispatch(cmd,group_x,group_y,1,{set},&push,sizeof(push));
            compute_barrier(cmd);
            filter.dispatch(cmd,group_x,group_y,1,{set},&push,sizeof(push));
            image_barrier(cmd,display,
                vk::PipelineStageFlagBits::eComputeShader,
                vk::AccessFlagBits::eShaderWrite,
                vk::PipelineStageFlagBits::eFragmentShader,
                vk::AccessFlagBits::eShaderRead,
                vk::ImageLayout::eGeneral,vk::ImageLayout::eGeneral);
            image_barrier(cmd,positions[current],
                vk::PipelineStageFlagBits::eComputeShader,
                vk::AccessFlagBits::eShaderWrite,
                vk::PipelineStageFlagBits::eFragmentShader,
                vk::AccessFlagBits::eShaderRead,
                vk::ImageLayout::eGeneral,vk::ImageLayout::eGeneral);
            valid_history=true;
            have_output=true;
            last_eye=eye;last_right=right;last_up=up;
            ++frame;
        }
        image_output output() const
        {
            if(!have_output)return {};
            return {display.handle(),display.view(),extent,vk::ImageLayout::eGeneral};
        }
        image_output hit_positions() const
        {
            if(!have_output)return {};
            return {positions[current].handle(),positions[current].view(),
                    extent,vk::ImageLayout::eGeneral};
        }
        progress status() const{return {frame,stable,moved};}
        void invalidate(){valid_history=false;stable=0;}
    };

    tracer::tracer()=default;
    tracer::~tracer()=default;
    void tracer::create(const scene& s,vk::Extent2D e,
                        const configuration& cfg)
    {
        state_.reset();
        auto next=std::make_unique<implementation>();
        next->create(s,e,cfg);
        state_=std::move(next);
    }
    void tracer::resize(vk::Extent2D e)
    {
        if(!state_)throw std::logic_error("pt: not created");
        scene world=state_->world;
        configuration config=state_->cfg;
        create(world,e,config);
    }
    void tracer::set_scene(const scene& s)
    {
        if(!state_)throw std::logic_error("pt: not created");
        vk::Extent2D extent=state_->extent;
        configuration config=state_->cfg;
        create(s,extent,config);
    }
    void tracer::invalidate()
    {
        if(!state_)throw std::logic_error("pt: not created");
        state_->invalidate();
    }
    image_output tracer::record(vk::CommandBuffer cmd,const view& camera)
    {
        if(!state_)throw std::logic_error("pt: not created");
        state_->record(cmd,camera);
        return state_->output();
    }
    image_output tracer::output() const
    {
        return state_?state_->output():image_output{};
    }
    image_output tracer::surface_positions() const
    {
        return state_?state_->hit_positions():image_output{};
    }
    progress tracer::status() const
    {
        return state_?state_->status():progress{};
    }
    void tracer::destroy(){state_.reset();}
}
