#include "continuum.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace fld::continuum
{
    namespace
    {
        struct simulation_push
        {
            vec4 origin_cell,spawn_radius,physical,interface_data;
            std::array<std::uint32_t,4> dims,flags;
        };
        static_assert(sizeof(simulation_push)==96);
        struct render_push
        {
            mat4 inverse_vp;
            vec4 origin_cell,tint;
            std::array<std::uint32_t,4> grid,screen;
        };
        static_assert(sizeof(render_push)==128);

        struct frame
        {
            vk::DescriptorSet descriptors{};
            vk::image::resource low{};
            vk::ImageView low_view{};
            bool transitioned=false;
        };
        struct state
        {
            options settings{};
            drop initial{};
            vk::buffer::resource fields,boxes;
            vk::DescriptorSetLayout grid_layout{},render_layout{};
            vk::DescriptorPool grid_pool{},render_pool{};
            vk::DescriptorSet grid_set{};
            vk::PipelineLayout grid_pipeline_layout{},render_pipeline_layout{};
            std::array<vk::Pipeline,7> grid_pipelines{};
            std::array<vk::Pipeline,2> render_pipelines{};
            std::vector<frame> frames;
            std::uint32_t screen_width=0,screen_height=0;
            std::uint32_t low_width=0,low_height=0,active_vof=0;
            std::uint32_t support_layer=0; // 0 disables thin-ground film
            bool seeded=false;
        };
        state* data=nullptr;

        void barrier(vk::CommandBuffer cmd)
        {
            vk::MemoryBarrier memory{};
            memory.srcAccessMask=vk::AccessFlagBits::eShaderWrite;
            memory.dstAccessMask=vk::AccessFlagBits::eShaderRead |
                                 vk::AccessFlagBits::eShaderWrite;
            cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                vk::PipelineStageFlagBits::eComputeShader,
                                {},{memory},{},{});
        }

        mat4 inverse(const mat4& m)
        {
            double a[4][8]{};
            for(int row=0;row<4;++row)
            {
                for(int col=0;col<4;++col)a[row][col]=m(row,col);
                a[row][row+4]=1.;
            }
            for(int col=0;col<4;++col)
            {
                int pivot=col;
                for(int row=col+1;row<4;++row)
                    if(std::abs(a[row][col])>std::abs(a[pivot][col]))
                        pivot=row;
                if(!std::isfinite(a[pivot][col]) ||
                   std::abs(a[pivot][col])<1e-12)
                    throw std::invalid_argument("Singular fluid view matrix");
                for(int k=0;k<8;++k)std::swap(a[pivot][k],a[col][k]);
                const double scale=a[col][col];
                for(int k=0;k<8;++k)a[col][k]/=scale;
                for(int row=0;row<4;++row)
                {
                    if(row==col)continue;
                    const double factor=a[row][col];
                    for(int k=0;k<8;++k)a[row][k]-=factor*a[col][k];
                }
            }
            mat4 result{};
            for(int row=0;row<4;++row)
                for(int col=0;col<4;++col)
                    result(row,col)=float(a[row][col+4]);
            return result;
        }

        void dispatch_grid(vk::CommandBuffer cmd,std::uint32_t stage,
                           const simulation_push& push)
        {
            const auto& o=data->settings;
            vk::record::pipeline(cmd,vk::PipelineBindPoint::eCompute,
                                 data->grid_pipelines[stage]);
            cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                   data->grid_pipeline_layout,0,
                                   {data->grid_set},{});
            cmd.pushConstants(data->grid_pipeline_layout,
                              vk::ShaderStageFlagBits::eCompute,
                              0,sizeof(push),&push);
            vk::record::dispatch(cmd,(o.width+8)/8,
                                 (o.height+8)/8,(o.depth+4)/4);
            barrier(cmd);
        }

        void dispatch_render(vk::CommandBuffer cmd,std::uint32_t stage,
                             const frame& f,const render_push& push)
        {
            vk::record::pipeline(cmd,vk::PipelineBindPoint::eCompute,
                                 data->render_pipelines[stage]);
            cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                   data->render_pipeline_layout,0,
                                   {f.descriptors},{});
            cmd.pushConstants(data->render_pipeline_layout,
                              vk::ShaderStageFlagBits::eCompute,
                              0,sizeof(push),&push);
            vk::record::dispatch(cmd,(push.screen[0]+7)/8,
                                 (push.screen[1]+7)/8,1);
            barrier(cmd);
        }

        vk::Pipeline make_pipeline(const std::string& path,
                                   vk::PipelineLayout layout)
        {
            const auto module=vk::shader::create(vk::shader::read(path));
            try
            {
                vk::ComputePipelineCreateInfo info{};
                info.stage.stage=vk::ShaderStageFlagBits::eCompute;
                info.stage.module=module;
                info.stage.pName="main";
                info.layout=layout;
                auto pipeline=vk::pipeline::compute(info);
                vk::shader::destroy(module);
                return pipeline;
            }
            catch(...)
            {
                vk::shader::destroy(module);
                throw;
            }
        }

        vk::DescriptorSetLayout make_layout(
            const std::vector<std::pair<std::uint32_t,vk::DescriptorType>>& entries)
        {
            std::vector<vk::DescriptorSetLayoutBinding> bindings(entries.size());
            for(std::size_t i=0;i<entries.size();++i)
            {
                bindings[i].binding=entries[i].first;
                bindings[i].descriptorType=entries[i].second;
                bindings[i].descriptorCount=1;
                bindings[i].stageFlags=vk::ShaderStageFlagBits::eCompute;
            }
            vk::DescriptorSetLayoutCreateInfo info{};
            info.bindingCount=std::uint32_t(bindings.size());
            info.pBindings=bindings.data();
            return vk::descriptor_layout::create(info);
        }

        vk::PipelineLayout make_pipeline_layout(vk::DescriptorSetLayout layout,
                                                std::uint32_t push_size)
        {
            vk::PushConstantRange range{};
            range.stageFlags=vk::ShaderStageFlagBits::eCompute;
            range.size=push_size;
            vk::PipelineLayoutCreateInfo info{};
            info.setLayoutCount=1;
            info.pSetLayouts=&layout;
            info.pushConstantRangeCount=1;
            info.pPushConstantRanges=&range;
            return vk::pipeline_layout::create(info);
        }

        simulation_push constants(float dt)
        {
            const auto& cfg=data->settings;
            simulation_push p{};
            p.origin_cell={cfg.origin.x,cfg.origin.y,cfg.origin.z,cfg.cell_size};
            p.spawn_radius={data->initial.center.x,data->initial.center.y,
                            data->initial.center.z,data->initial.radius};
            p.physical={dt,cfg.liquid.density,cfg.gravity.y,
                        cfg.liquid.dynamic_viscosity/cfg.liquid.density};
            p.interface_data={cfg.liquid.surface_tension,
                              cfg.liquid.advancing_angle,0.f,0.f};
            p.dims={cfg.width,cfg.height,cfg.depth,
                    std::uint32_t(cfg.colliders.size())};
            p.flags={data->active_vof,0,0,0};
            return p;
        }
    }

    bool active(){return data && data->seeded;}

    void init(const options& settings,const drop& initial,
              const shader_paths& shaders,std::uint32_t frames,
              std::uint32_t screen_width,std::uint32_t screen_height)
    {
        if(data)throw std::logic_error("Continuum liquid already initialized");
        if(!valid(settings.liquid) || !frames || !screen_width || !screen_height ||
           !settings.width || !settings.height || !settings.depth ||
           settings.width>128 || settings.height>128 || settings.depth>128 ||
           settings.colliders.size()>1024 ||
           !std::isfinite(settings.cell_size) || settings.cell_size<=0.f ||
           !std::isfinite(settings.render_scale) ||
           settings.render_scale<.25f || settings.render_scale>1.f ||
           !std::isfinite(initial.radius) || initial.radius<=settings.cell_size ||
           !std::isfinite(settings.gravity.y) ||
           !settings.pressure_iterations || settings.pressure_iterations>128 ||
           !settings.substeps || settings.substeps>8 ||
           !vk::device::get()->handle)
            throw std::invalid_argument("Invalid continuum fluid settings");
        for(const auto& b:settings.colliders)
            if(!std::isfinite(b.minimum.x) || !std::isfinite(b.minimum.y) ||
               !std::isfinite(b.minimum.z) || !std::isfinite(b.maximum.x) ||
               !std::isfinite(b.maximum.y) || !std::isfinite(b.maximum.z) ||
               b.minimum.x>=b.maximum.x || b.minimum.y>=b.maximum.y ||
               b.minimum.z>=b.maximum.z)
                throw std::invalid_argument("Invalid fluid collision bounds");
        data=new state{};
        data->settings=settings;
        data->initial=initial;
        data->screen_width=screen_width;
        data->screen_height=screen_height;
        data->low_width=std::max(1u,std::uint32_t(std::ceil(
            float(screen_width)*settings.render_scale)));
        data->low_height=std::max(1u,std::uint32_t(std::ceil(
            float(screen_height)*settings.render_scale)));
        // Identify the actual upper face of a scene-owned solid collider
        // below the drop. Only use aligned grid faces for thin-film rendering.
        // Physics always uses the full voxelized collider field independently.
        float support=-std::numeric_limits<float>::infinity();
        for(const auto& collider:settings.colliders)
        {
            if(initial.center.x<collider.minimum.x ||
               initial.center.x>collider.maximum.x ||
               initial.center.z<collider.minimum.z ||
               initial.center.z>collider.maximum.z ||
               collider.maximum.y>initial.center.y ||
               collider.maximum.y<=support)continue;
            support=collider.maximum.y;
        }
        if(std::isfinite(support))
        {
            const float layer=(support-settings.origin.y)/settings.cell_size;
            const float rounded=std::round(layer);
            if(std::abs(layer-rounded)<.01f && rounded>=1.f &&
               rounded<float(settings.height))
                data->support_layer=std::uint32_t(rounded);
        }
        auto device=vk::device::get()->handle;
        try
        {
            const auto n=std::size_t(settings.width)*settings.height*settings.depth;
            const auto u=std::size_t(settings.width+1)*settings.height*settings.depth;
            const auto v=std::size_t(settings.width)*(settings.height+1)*settings.depth;
            const auto w=std::size_t(settings.width)*settings.height*(settings.depth+1);
            const auto storage=(7*n+2*(u+v+w))*sizeof(float);
            data->fields=vk::buffer::create(storage,
                vk::BufferUsageFlagBits::eStorageBuffer,
                vk::MemoryPropertyFlagBits::eDeviceLocal);
            data->boxes=vk::buffer::create(
                std::max<std::size_t>(1,settings.colliders.size())*sizeof(box),
                vk::BufferUsageFlagBits::eStorageBuffer,
                vk::MemoryPropertyFlagBits::eHostVisible |
                vk::MemoryPropertyFlagBits::eHostCoherent);
            if(!settings.colliders.empty())
                vk::buffer::write(data->boxes,settings.colliders.data(),
                                  settings.colliders.size()*sizeof(box));

            data->grid_layout=make_layout({
                {0,vk::DescriptorType::eStorageBuffer},
                {1,vk::DescriptorType::eStorageBuffer}
            });
            data->grid_pipeline_layout=make_pipeline_layout(
                data->grid_layout,sizeof(simulation_push));
            for(std::size_t i=0;i<data->grid_pipelines.size();++i)
                data->grid_pipelines[i]=make_pipeline(shaders.simulation[i],
                                                     data->grid_pipeline_layout);

            vk::DescriptorPoolSize grid_pool_size{
                vk::DescriptorType::eStorageBuffer,2};
            vk::DescriptorPoolCreateInfo grid_pool_info{};
            grid_pool_info.maxSets=1;
            grid_pool_info.poolSizeCount=1;
            grid_pool_info.pPoolSizes=&grid_pool_size;
            data->grid_pool=device.createDescriptorPool(grid_pool_info);
            vk::DescriptorSetAllocateInfo allocation{};
            allocation.descriptorPool=data->grid_pool;
            allocation.descriptorSetCount=1;
            allocation.pSetLayouts=&data->grid_layout;
            data->grid_set=device.allocateDescriptorSets(allocation).front();
            const std::array<vk::DescriptorBufferInfo,2> buffers{{
                vk::buffer::descriptor(data->fields),
                vk::buffer::descriptor(data->boxes)
            }};
            std::array<vk::WriteDescriptorSet,2> writes{};
            for(std::uint32_t i=0;i<2;++i)
            {
                writes[i].dstSet=data->grid_set;
                writes[i].dstBinding=i;
                writes[i].descriptorType=vk::DescriptorType::eStorageBuffer;
                writes[i].descriptorCount=1;
                writes[i].pBufferInfo=&buffers[i];
            }
            device.updateDescriptorSets(writes,{});

            data->render_layout=make_layout({
                {0,vk::DescriptorType::eCombinedImageSampler},
                {1,vk::DescriptorType::eCombinedImageSampler},
                {2,vk::DescriptorType::eStorageImage},
                {3,vk::DescriptorType::eStorageBuffer},
                {6,vk::DescriptorType::eStorageImage}
            });
            data->render_pipeline_layout=make_pipeline_layout(
                data->render_layout,sizeof(render_push));
            for(std::size_t i=0;i<data->render_pipelines.size();++i)
                data->render_pipelines[i]=make_pipeline(shaders.rendering[i],
                                                       data->render_pipeline_layout);
            const std::array<vk::DescriptorPoolSize,3> sizes{{
                {vk::DescriptorType::eCombinedImageSampler,frames*2},
                {vk::DescriptorType::eStorageImage,frames*2},
                {vk::DescriptorType::eStorageBuffer,frames}
            }};
            vk::DescriptorPoolCreateInfo render_pool_info{};
            render_pool_info.maxSets=frames;
            render_pool_info.poolSizeCount=sizes.size();
            render_pool_info.pPoolSizes=sizes.data();
            data->render_pool=device.createDescriptorPool(render_pool_info);
            std::vector<vk::DescriptorSetLayout> layouts(frames,
                                                         data->render_layout);
            allocation.descriptorPool=data->render_pool;
            allocation.descriptorSetCount=frames;
            allocation.pSetLayouts=layouts.data();
            const auto sets=device.allocateDescriptorSets(allocation);
            data->frames.resize(frames);
            for(std::uint32_t i=0;i<frames;++i)
            {
                auto& f=data->frames[i];
                f.descriptors=sets[i];
                vk::ImageCreateInfo info{};
                info.imageType=vk::ImageType::e2D;
                info.format=vk::Format::eR16G16B16A16Sfloat;
                info.extent=vk::Extent3D{data->low_width,data->low_height,1};
                info.mipLevels=1;
                info.arrayLayers=1;
                info.samples=vk::SampleCountFlagBits::e1;
                info.tiling=vk::ImageTiling::eOptimal;
                info.usage=vk::ImageUsageFlagBits::eStorage;
                f.low=vk::image::create(info,vk::MemoryPropertyFlagBits::eDeviceLocal);
                vk::ImageViewCreateInfo view{};
                view.image=f.low.handle;
                view.viewType=vk::ImageViewType::e2D;
                view.format=info.format;
                view.subresourceRange={vk::ImageAspectFlagBits::eColor,0,1,0,1};
                f.low_view=vk::image_view::create(view);
            }
        }
        catch(...)
        {
            shutdown();
            throw;
        }
    }

    void record(vk::CommandBuffer command,std::uint32_t image,
                float elapsed_seconds,const render_input& target)
    {
        if(!data || !command || image>=data->frames.size() ||
           !target.depth || !target.scene_color || !target.output ||
           !target.sampler ||
           target.width!=data->screen_width ||
           target.height!=data->screen_height ||
           !std::isfinite(elapsed_seconds) || elapsed_seconds<0.f)
            throw std::invalid_argument("Invalid continuum fluid frame");
        auto& f=data->frames[image];
        if(!f.transitioned)
        {
            vk::ImageMemoryBarrier change{};
            change.oldLayout=vk::ImageLayout::eUndefined;
            change.newLayout=vk::ImageLayout::eGeneral;
            change.srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
            change.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
            change.image=f.low.handle;
            change.subresourceRange={vk::ImageAspectFlagBits::eColor,0,1,0,1};
            change.dstAccessMask=vk::AccessFlagBits::eShaderWrite;
            command.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
                                    vk::PipelineStageFlagBits::eComputeShader,
                                    {},{},{},{change});
            f.transitioned=true;
        }
        if(!data->seeded)
        {
            dispatch_grid(command,0,constants(0.f)); // seed VOF, solids and MAC faces
            data->seeded=true;
        }

        if(elapsed_seconds>0.f)
        {
            const auto& o=data->settings;
            // Bounded CFL input. Low frame rates slow simulated time rather
            // than permitting unbounded, nonconservative explicit fluxes.
            const float frame_seconds=std::min(elapsed_seconds,1.f/60.f);
            const float dt=frame_seconds/float(o.substeps);
            for(std::uint32_t sub=0;sub<o.substeps;++sub)
            {
                simulation_push p=constants(dt);
                dispatch_grid(command,1,p); // velocity advection + body force
                dispatch_grid(command,2,p); // surface curvature
                dispatch_grid(command,3,p); // divergence
                for(std::uint32_t iter=0;iter<o.pressure_iterations;++iter)
                {
                    p.flags[1]=iter&1u;
                    dispatch_grid(command,4,p); // Poisson Jacobi iteration
                }
                p.flags[1]=o.pressure_iterations&1u;
                dispatch_grid(command,5,p); // MAC face pressure projection
                dispatch_grid(command,6,p); // conservative volume flux
                data->active_vof^=1u;
            }
        }
        const std::array<vk::DescriptorImageInfo,4> images{{
            {target.sampler,target.depth,target.depth_layout},
            {target.sampler,target.scene_color,target.color_layout},
            {{},target.output,vk::ImageLayout::eGeneral},
            {{},f.low_view,vk::ImageLayout::eGeneral}
        }};
        const auto field_info=vk::buffer::descriptor(data->fields);
        const std::array<std::uint32_t,5> bindings{{0,1,2,3,6}};
        std::array<vk::WriteDescriptorSet,5> updates{};
        for(std::uint32_t i=0;i<updates.size();++i)
        {
            auto& update=updates[i];
            update.dstSet=f.descriptors;
            update.dstBinding=bindings[i];
            update.descriptorCount=1;
            if(i==3)
            {
                update.descriptorType=vk::DescriptorType::eStorageBuffer;
                update.pBufferInfo=&field_info;
            }
            else
            {
                update.descriptorType=(i==2 || i==4)?
                    vk::DescriptorType::eStorageImage:
                    vk::DescriptorType::eCombinedImageSampler;
                update.pImageInfo=&images[i==4?3:i];
            }
        }
        vk::device::get()->handle.updateDescriptorSets(updates,{});

        render_push rp{};
        rp.inverse_vp=inverse(target.view_projection);
        const auto& o=data->settings;
        rp.origin_cell={o.origin.x,o.origin.y,o.origin.z,o.cell_size};
        rp.tint={std::clamp(target.debug_color.x,0.f,1.f),
                 std::clamp(target.debug_color.y,0.f,1.f),
                 std::clamp(target.debug_color.z,0.f,1.f),
                 std::clamp(target.debug_opacity,0.f,1.f)};
        rp.grid={o.width,o.height,o.depth,data->support_layer};
        rp.screen={data->low_width,data->low_height,
                   target.reversed_depth?1u:0u,data->active_vof};
        dispatch_render(command,0,f,rp);
        rp.screen[0]=data->screen_width;
        rp.screen[1]=data->screen_height;
        dispatch_render(command,1,f,rp);
    }

    void shutdown()
    {
        if(!data)return;
        const auto device=vk::device::get()->handle;
        if(device)
        {
            for(auto& f:data->frames)
            {
                if(f.low_view)vk::image_view::destroy(f.low_view);
                if(f.low.handle)vk::image::destroy(f.low);
            }
            for(auto& p:data->grid_pipelines)
                if(p)vk::pipeline::destroy(p);
            for(auto& p:data->render_pipelines)
                if(p)vk::pipeline::destroy(p);
            if(data->grid_pipeline_layout)
                vk::pipeline_layout::destroy(data->grid_pipeline_layout);
            if(data->render_pipeline_layout)
                vk::pipeline_layout::destroy(data->render_pipeline_layout);
            if(data->grid_pool)device.destroyDescriptorPool(data->grid_pool);
            if(data->render_pool)device.destroyDescriptorPool(data->render_pool);
            if(data->grid_layout)
                vk::descriptor_layout::destroy(data->grid_layout);
            if(data->render_layout)
                vk::descriptor_layout::destroy(data->render_layout);
            if(data->fields.handle)vk::buffer::destroy(data->fields);
            if(data->boxes.handle)vk::buffer::destroy(data->boxes);
        }
        delete data;
        data=nullptr;
    }
}
