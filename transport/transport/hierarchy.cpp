#include "hierarchy.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

namespace transport::hierarchy
{
    namespace
    {
        using vec3=vk::util::math::vec3;
        using vk::util::math::dot;
        using vk::util::math::cross;
        using vk::util::math::length;
        using vk::util::math::normalize;

        constexpr std::uint32_t invalid=UINT32_MAX;
        constexpr float pi=3.14159265358979323846f;
        vec3 xyz(vec4 v){return {v.x,v.y,v.z};}
        float component(vec3 v,int axis)
        {
            return axis==0?v.x:(axis==1?v.y:v.z);
        }

        struct alignas(16) gpu_node
        {
            vec4 center_area{};
            vec4 normal_spread{};
            vec4 extent_radius{};
            std::array<std::uint32_t,4> range_children{};
        };
        static_assert(sizeof(gpu_node)==64);

        struct link
        {
            std::uint32_t source=0;
            float factor=0.f;
        };
        static_assert(sizeof(link)==8);

        struct parameters
        {
            std::uint32_t patches=0,nodes=0,unused0=0,unused1=0;
        };
        enum kernel : std::uint32_t
        {
            seed=0,aggregate=1,gather=2,emission_aggregate=3,
            direct=4,inject=5,propagate=6,residual_kernel=7
        };
        constexpr std::array<const char*,8> kernel_names{{
            "seed","aggregate","gather","emission","direct","inject",
            "propagate","residual"
        }};

        vk::buffer::resource make_buffer(std::size_t bytes,
                                          vk::BufferUsageFlags usage,
                                          bool host)
        {
            const auto flags=host?
                (vk::MemoryPropertyFlagBits::eHostVisible |
                 vk::MemoryPropertyFlagBits::eHostCoherent):
                vk::MemoryPropertyFlags(vk::MemoryPropertyFlagBits::eDeviceLocal);
            return vk::buffer::create(std::max<std::size_t>(bytes,16),usage,flags);
        }

        template<class T>
        vk::buffer::resource upload(const std::vector<T>& contents)
        {
            auto b=make_buffer(contents.size()*sizeof(T),
                               vk::BufferUsageFlagBits::eStorageBuffer,true);
            if(!contents.empty())
                vk::buffer::write(b,contents.data(),contents.size()*sizeof(T));
            return b;
        }

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

        vec3 quad_sample(const surface& s,int index)
        {
            const float u=(float(index&1)+.5f)/2.f-.5f;
            const float v=(float(index>>1)+.5f)/2.f-.5f;
            return xyz(s.value.position_area)+xyz(s.tangent_u)*u+
                   xyz(s.tangent_v)*v;
        }

        bool occluded(vec3 a,vec3 b,
                      const std::vector<transport::oriented_blocker>& boxes)
        {
            const vec3 ray=b-a;
            for(const auto& box:boxes)
            {
                const vec3 rel=a-xyz(box.center);
                const vec3 axes[]{xyz(box.axis_x),xyz(box.axis_y),xyz(box.axis_z)};
                const vec3 pos{dot(rel,axes[0]),dot(rel,axes[1]),dot(rel,axes[2])};
                const vec3 dir{dot(ray,axes[0]),dot(ray,axes[1]),dot(ray,axes[2])};
                const vec3 extent=xyz(box.half_extent);
                float low=.001f,high=.999f;
                bool separated=false;
                for(int i=0;i<3;i++)
                {
                    const float p=component(pos,i),d=component(dir,i),
                                e=component(extent,i);
                    if(std::abs(d)<1e-7f)
                    {
                        if(std::abs(p)>e){separated=true;break;}
                    }
                    else
                    {
                        const float t0=(-e-p)/d,t1=(e-p)/d;
                        low=std::max(low,std::min(t0,t1));
                        high=std::min(high,std::max(t0,t1));
                        if(low>high){separated=true;break;}
                    }
                }
                if(!separated && high>=low)return true;
            }
            return false;
        }

        // Conservative broad-phase for an entire source cluster. If a
        // blocker intersects the segment swept by its bounding sphere, we
        // must split instead of replacing partial visibility by a binary
        // three-ray decision.
        bool maybe_blocked(vec3 a,vec3 b,float radius,
                           const std::vector<transport::oriented_blocker>& boxes)
        {
            const vec3 ray=b-a;
            for(const auto& box:boxes)
            {
                const vec3 axes[]{xyz(box.axis_x),xyz(box.axis_y),xyz(box.axis_z)};
                const vec3 start=a-xyz(box.center);
                const vec3 p{dot(start,axes[0]),dot(start,axes[1]),
                             dot(start,axes[2])};
                const vec3 d{dot(ray,axes[0]),dot(ray,axes[1]),dot(ray,axes[2])};
                const vec3 extent=xyz(box.half_extent);
                float lo=0.f,hi=1.f;
                bool separated=false;
                for(int axis=0;axis<3;axis++)
                {
                    const float x=component(p,axis),v=component(d,axis);
                    const float half=component(extent,axis)+radius;
                    if(std::abs(v)<1e-8f)
                    {
                        if(std::abs(x)>half){separated=true;break;}
                    }
                    else
                    {
                        const float t0=(-half-x)/v,t1=(half-x)/v;
                        lo=std::max(lo,std::min(t0,t1));
                        hi=std::min(hi,std::max(t0,t1));
                        if(lo>hi){separated=true;break;}
                    }
                }
                if(!separated && hi>=lo)return true;
            }
            return false;
        }

        float kernel(vec3 a,vec3 na,vec3 b,vec3 nb,
                     const std::vector<transport::oriented_blocker>& boxes)
        {
            const vec3 d=b-a;
            const float r2=dot(d,d);
            if(r2<=1e-8f)return 0.f;
            const float inv=1.f/std::sqrt(r2);
            const float ca=std::max(0.f,dot(na,d)*inv);
            const float cb=std::max(0.f,-dot(nb,d)*inv);
            if(ca*cb==0.f)return 0.f;
            if(occluded(a+na*.001f,b+nb*.001f,boxes))return 0.f;
            return ca*cb/(pi*r2);
        }

        float leaf_form_factor(const surface& receiver,const surface& sender,
                               const std::vector<transport::oriented_blocker>& boxes)
        {
            const vec3 na=normalize(xyz(receiver.value.normal));
            const vec3 nb=normalize(xyz(sender.value.normal));
            float sum=0.f;
            for(int a=0;a<4;a++)
            for(int b=0;b<4;b++)
                sum+=kernel(quad_sample(receiver,a),na,
                            quad_sample(sender,b),nb,boxes);
            return sender.value.position_area.w*sum/16.f;
        }

        struct graph_builder
        {
            const std::vector<surface>& surface_data;
            const std::vector<transport::oriented_blocker>& boxes;
            options config;
            std::vector<gpu_node> nodes;
            std::vector<std::uint32_t> order,rank,offsets;
            std::vector<link> links;
            float maximum_row=0.f,global_scale=1.f;

            graph_builder(const std::vector<surface>& s,
                          const std::vector<transport::oriented_blocker>& o,
                          options cfg):surface_data(s),boxes(o),config(cfg)
            {
                order.resize(s.size());
                rank.resize(s.size());
                std::iota(order.begin(),order.end(),0u);
                nodes.reserve(2*s.size());
                offsets.reserve(s.size()+1);
            }

            std::uint32_t build_tree(std::uint32_t first,std::uint32_t last)
            {
                const auto id=std::uint32_t(nodes.size());
                nodes.emplace_back();
                vec3 center{},normal_sum{};
                float area=0.f;
                const float inf=std::numeric_limits<float>::max();
                vec3 minimum{inf,inf,inf},maximum{-inf,-inf,-inf};
                for(std::uint32_t i=first;i<last;i++)
                {
                    const auto& p=surface_data[order[i]].value;
                    const vec3 pos=xyz(p.position_area);
                    const float a=p.position_area.w;
                    center=center+pos*a;
                    normal_sum=normal_sum+xyz(p.normal)*a;
                    area+=a;
                    minimum.x=std::min(minimum.x,pos.x);
                    minimum.y=std::min(minimum.y,pos.y);
                    minimum.z=std::min(minimum.z,pos.z);
                    maximum.x=std::max(maximum.x,pos.x);
                    maximum.y=std::max(maximum.y,pos.y);
                    maximum.z=std::max(maximum.z,pos.z);
                }
                if(area<=0.f)throw std::runtime_error("invalid cluster area");
                center=center/area;
                const vec3 normal=normalize(normal_sum);
                float spread=1.f,radius=0.f;
                for(std::uint32_t i=first;i<last;i++)
                {
                    const auto& p=surface_data[order[i]].value;
                    spread=std::min(spread,dot(normal,normalize(xyz(p.normal))));
                    const auto& geometry=surface_data[order[i]];
                    const float extent=.5f*(
                        length(xyz(geometry.tangent_u))+
                        length(xyz(geometry.tangent_v)));
                    radius=std::max(radius,length(xyz(p.position_area)-center)+
                                    extent);
                }
                nodes[id].center_area=vec4(center,area);
                nodes[id].normal_spread=vec4(normal,spread);
                nodes[id].extent_radius={0,0,0,radius};
                nodes[id].range_children={first,last-first,invalid,invalid};

                if(last-first>1)
                {
                    const vec3 span=maximum-minimum;
                    const int axis=span.x>=span.y&&span.x>=span.z?0:
                                   (span.y>=span.z?1:2);
                    const auto mid=first+(last-first)/2;
                    std::nth_element(order.begin()+first,order.begin()+mid,
                                     order.begin()+last,[&](auto a,auto b)
                    {
                        const auto pa=xyz(surface_data[a].value.position_area);
                        const auto pb=xyz(surface_data[b].value.position_area);
                        return component(pa,axis)<component(pb,axis);
                    });
                    const auto left=build_tree(first,mid);
                    const auto right=build_tree(mid,last);
                    nodes[id].range_children[2]=left;
                    nodes[id].range_children[3]=right;
                }
                return id;
            }

            void walk(std::uint32_t receiver,std::uint32_t source)
            {
                const auto& node=nodes[source];
                const auto first=node.range_children[0];
                const auto count=node.range_children[1];
                if(count==1 && order[first]==receiver)return;
                const auto& patch=surface_data[receiver];
                const vec3 pa=xyz(patch.value.position_area);
                const vec3 na=normalize(xyz(patch.value.normal));
                const vec3 pb=xyz(node.center_area);
                const vec3 nb=xyz(node.normal_spread);
                const float distance=length(pb-pa);
                const float radius=node.extent_radius.w+
                                   .71f*std::sqrt(patch.value.position_area.w);
                const bool self=rank[receiver]>=first&&rank[receiver]<first+count;
                const bool distant=distance>1e-5f &&
                    radius/distance<config.far_field_ratio;
                const bool coherent=node.normal_spread.w>=config.normal_threshold;
                // Only collapse a cluster if visibility is provably clear
                // under a conservative source-region bound. This cannot
                // silently drop partially illuminated clusters.
                const bool clear=!maybe_blocked(
                    pa+na*.001f,pb,
                    node.extent_radius.w+
                        .5f*(length(xyz(patch.tangent_u))+
                             length(xyz(patch.tangent_v))),
                    boxes);
                if(count>1 && (self || !distant || !coherent || !clear))
                {
                    walk(receiver,node.range_children[2]);
                    walk(receiver,node.range_children[3]);
                    return;
                }
                float weight=count==1?
                    leaf_form_factor(patch,surface_data[order[first]],boxes):
                    node.center_area.w*kernel(pa,na,pb,nb,boxes);
                if(weight>1e-10f)
                {
                    if(links.size()>=config.max_edges)
                        throw std::runtime_error("hierarchy link budget exceeded");
                    links.push_back({source,weight});
                }
            }

            void build()
            {
                build_tree(0,std::uint32_t(order.size()));
                for(std::uint32_t i=0;i<order.size();i++)rank[order[i]]=i;
                for(std::uint32_t i=0;i<surface_data.size();i++)
                {
                    offsets.push_back(std::uint32_t(links.size()));
                    walk(i,0);
                    float total=0.f;
                    for(std::uint32_t j=offsets.back();j<links.size();j++)
                        total+=links[j].factor;
                    maximum_row=std::max(maximum_row,total);
                }
                offsets.push_back(std::uint32_t(links.size()));
                // Global scaling preserves near-pair reciprocity but is a
                // biased safety clamp. Track the scale for accuracy reporting.
                global_scale=maximum_row>1.f?1.f/maximum_row:1.f;
                if(global_scale<1.f)
                    for(auto& edge:links)edge.factor*=global_scale;
            }
        };
    }

    struct solver::state
    {
        statistics info{};
        std::vector<surface> surfaces;
        vk::buffer::resource patches,nodes,order,offsets,links,history;
        vk::buffer::resource oriented,readback;
        std::array<vk::buffer::resource,2> radiance,delta;
        vk::buffer::resource node_values,first,residual_value;
        vk::DescriptorSetLayout descriptor_layout{};
        vk::DescriptorPool pool{};
        std::array<vk::DescriptorSet,4> sets{};
        vk::PipelineLayout pipeline_layout{};
        std::array<vk::Pipeline,8> pipelines{};
    };

    solver::solver()=default;

    solver::~solver(){shutdown();}

    void solver::init(const std::vector<surface>& surfaces,
                      const std::vector<transport::oriented_blocker>& boxes,
                      const std::string& shader_prefix,options config)
    {
        if(state_)throw std::logic_error("hierarchy already initialized");
        if(!vk::device::get()->handle || surfaces.empty() ||
           surfaces.size()>100000 || config.max_edges==0 ||
           !(config.far_field_ratio>0.f&&config.far_field_ratio<=1.f))
            throw std::invalid_argument("invalid hierarchy initialization");
        for(const auto& s:surfaces)
        {
            const auto& p=s.value;
            const vec4 values[]{p.position_area,p.normal,p.albedo,p.emission,
                                s.tangent_u,s.tangent_v};
            for(const auto& v:values)
                for(const float x:{v.x,v.y,v.z,v.w})
                    if(!std::isfinite(x))throw std::invalid_argument("nonfinite patch");
            if(p.position_area.w<=0.f || length(xyz(p.normal))<.5f)
                throw std::invalid_argument("invalid area/normal");
            for(const float x:{p.albedo.x,p.albedo.y,p.albedo.z})
                if(x<0.f||x>=1.f)throw std::invalid_argument("invalid albedo");
        }

        state_=std::make_unique<state>();
        auto& s=*state_;
        s.surfaces=surfaces;
        try
        {
            const auto t0=std::chrono::steady_clock::now();
            graph_builder graph(surfaces,boxes,config);
            graph.build();
            s.info.surface_count=std::uint32_t(surfaces.size());
            s.info.node_count=std::uint32_t(graph.nodes.size());
            s.info.edge_count=std::uint32_t(graph.links.size());
            s.info.initial_max_row=graph.maximum_row;
             s.info.global_scale=graph.global_scale;
            float maximum_reflectance=0.f;
            for(const auto& patch:surfaces)
                maximum_reflectance=std::max({
                    maximum_reflectance,
                    patch.value.albedo.x,patch.value.albedo.y,
                    patch.value.albedo.z
                });
            s.info.operator_norm_upper=maximum_reflectance*
                graph.maximum_row*graph.global_scale;
            // Validate reciprocity of near-field finite-area leaf links.
            // Clustered links are approximations and are not included here.
            std::unordered_map<std::uint64_t,float> flux;
            flux.reserve(graph.links.size());
            for(std::uint32_t i=0;i<surfaces.size();++i)
                for(std::uint32_t e=graph.offsets[i];e<graph.offsets[i+1];++e)
                {
                    const auto& l=graph.links[e];
                    const auto& n=graph.nodes[l.source];
                    if(n.range_children[1]!=1)continue;
                    const auto j=graph.order[n.range_children[0]];
                    const std::uint64_t key=(std::uint64_t(i)<<32)|j;
                    flux.emplace(key,l.factor*surfaces[i].value.position_area.w);
                }
            for(const auto& [key,value]:flux)
            {
                const std::uint32_t i=std::uint32_t(key>>32);
                const std::uint32_t j=std::uint32_t(key);
                if(j<=i)continue;
                const std::uint64_t inverse=(std::uint64_t(j)<<32)|i;
                const auto found=flux.find(inverse);
                if(found==flux.end())continue;
                const float difference=std::abs(value-found->second)/
                    std::max({std::abs(value),std::abs(found->second),1e-10f});
                s.info.reciprocity_max_error=std::max(s.info.reciprocity_max_error,
                                                     difference);
                s.info.reciprocal_pairs++;
            }
            s.info.graph_build_ms=float(std::chrono::duration<double,std::milli>(
                std::chrono::steady_clock::now()-t0).count());

            std::vector<transport::patch> patch_data;
            std::vector<vec4> emission_data;
            patch_data.reserve(surfaces.size());
            emission_data.reserve(surfaces.size());
            for(const auto& v:surfaces)
            {
                patch_data.push_back(v.value);
                emission_data.push_back(v.value.emission);
            }
            s.patches=upload(patch_data);
            s.nodes=upload(graph.nodes);
            s.order=upload(graph.order);
            s.offsets=upload(graph.offsets);
            s.links=upload(graph.links);
            s.history=upload(emission_data);
            s.oriented=upload(boxes);
            const auto bytes=surfaces.size()*sizeof(vec4);
            for(auto& b:s.radiance)
                b=make_buffer(bytes,vk::BufferUsageFlagBits::eStorageBuffer,false);
            for(auto& b:s.delta)
                b=make_buffer(bytes,vk::BufferUsageFlagBits::eStorageBuffer,false);
            s.node_values=make_buffer(graph.nodes.size()*sizeof(vec4),
                          vk::BufferUsageFlagBits::eStorageBuffer,false);
            s.first=make_buffer(bytes,vk::BufferUsageFlagBits::eStorageBuffer,false);
            s.residual_value=make_buffer(16,
                          vk::BufferUsageFlagBits::eStorageBuffer |
                          vk::BufferUsageFlagBits::eTransferDst |
                          vk::BufferUsageFlagBits::eTransferSrc,false);
            s.readback=make_buffer(16,vk::BufferUsageFlagBits::eTransferDst,true);

            const auto dev=vk::device::get()->handle;
            std::array<vk::DescriptorSetLayoutBinding,12> bindings{};
            for(std::uint32_t i=0;i<bindings.size();i++)
            {
                bindings[i].binding=i;
                bindings[i].descriptorType=vk::DescriptorType::eStorageBuffer;
                bindings[i].descriptorCount=1;
                bindings[i].stageFlags=vk::ShaderStageFlagBits::eCompute;
            }
            vk::DescriptorSetLayoutCreateInfo descriptor_layout{};
            descriptor_layout.bindingCount=std::uint32_t(bindings.size());
            descriptor_layout.pBindings=bindings.data();
            s.descriptor_layout=vk::descriptor_layout::create(descriptor_layout);
            vk::PushConstantRange range{};
            range.stageFlags=vk::ShaderStageFlagBits::eCompute;
            range.size=sizeof(parameters);
            vk::PipelineLayoutCreateInfo pipeline_layout{};
            pipeline_layout.setLayoutCount=1;
            pipeline_layout.pSetLayouts=&s.descriptor_layout;
            pipeline_layout.pushConstantRangeCount=1;
            pipeline_layout.pPushConstantRanges=&range;
            s.pipeline_layout=vk::pipeline_layout::create(pipeline_layout);

            for(std::size_t i=0;i<s.pipelines.size();i++)
            {
                const auto code=vk::shader::read(shader_prefix+kernel_names[i]+".spv");
                auto module=vk::shader::create(code);
                try
                {
                    vk::ComputePipelineCreateInfo info{};
                    info.stage.stage=vk::ShaderStageFlagBits::eCompute;
                    info.stage.module=module;
                    info.stage.pName="main";
                    info.layout=s.pipeline_layout;
                    s.pipelines[i]=vk::pipeline::compute(info);
                }
                catch(...)
                {
                    vk::shader::destroy(module);
                    throw;
                }
                vk::shader::destroy(module);
            }

            vk::DescriptorPoolSize pool_size{vk::DescriptorType::eStorageBuffer,48};
            vk::DescriptorPoolCreateInfo pool{};
            pool.maxSets=4;
            pool.poolSizeCount=1;
            pool.pPoolSizes=&pool_size;
            s.pool=dev.createDescriptorPool(pool);
            std::array<vk::DescriptorSetLayout,4> layouts{
                s.descriptor_layout,s.descriptor_layout,
                s.descriptor_layout,s.descriptor_layout};
            vk::DescriptorSetAllocateInfo allocation{};
            allocation.descriptorPool=s.pool;
            allocation.descriptorSetCount=4;
            allocation.pSetLayouts=layouts.data();
            const auto sets=dev.allocateDescriptorSets(allocation);
            for(std::size_t i=0;i<4;i++)s.sets[i]=sets[i];

            for(std::uint32_t j=0;j<4;j++)
            {
                const auto& input=j<2?s.radiance[j]:s.delta[j-2];
                const auto& output=j<2?s.radiance[1-j]:s.delta[3-j];
                std::array<vk::DescriptorBufferInfo,12> infos{{
                    vk::buffer::descriptor(s.patches),
                    vk::buffer::descriptor(s.nodes),
                    vk::buffer::descriptor(s.order),
                    vk::buffer::descriptor(s.offsets),
                    vk::buffer::descriptor(s.links),
                    vk::buffer::descriptor(input),
                    vk::buffer::descriptor(output),
                    vk::buffer::descriptor(s.node_values),
                    vk::buffer::descriptor(s.first),
                    vk::buffer::descriptor(s.history),
                    vk::buffer::descriptor(s.radiance[0]),
                    vk::buffer::descriptor(s.residual_value)
                }};
                std::array<vk::WriteDescriptorSet,12> writes{};
                for(std::uint32_t i=0;i<12;i++)
                {
                    writes[i].dstSet=s.sets[j];
                    writes[i].dstBinding=i;
                    writes[i].descriptorCount=1;
                    writes[i].descriptorType=vk::DescriptorType::eStorageBuffer;
                    writes[i].pBufferInfo=&infos[i];
                }
                dev.updateDescriptorSets(writes,{});
            }
        }
        catch(...)
        {
            shutdown();
            throw;
        }
    }

    namespace
    {
        void dispatch(vk::CommandBuffer command,const solver::state& state,
                      std::uint32_t mode,std::uint32_t set)
        {
            vk::record::pipeline(command,vk::PipelineBindPoint::eCompute,
                                 state.pipelines[mode]);
            command.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                       state.pipeline_layout,0,{state.sets[set]},{});
            const parameters p{state.info.surface_count,state.info.node_count,0,0};
            command.pushConstants(state.pipeline_layout,
                vk::ShaderStageFlagBits::eCompute,0,sizeof(p),&p);
            const auto count=mode==aggregate||mode==emission_aggregate?
                state.info.node_count:state.info.surface_count;
            vk::record::dispatch(command,(count+127)/128,1,1);
            barrier(command);
        }
    }

    void solver::record_initial(vk::CommandBuffer command,std::uint32_t iterations)
    {
        if(!state_ || !command || iterations==0 || (iterations&1u) ||
           iterations>256)
            throw std::invalid_argument("invalid initial solve");
        const auto& s=*state_;
        dispatch(command,s,seed,1); // Set 1 output writes radiance[0].
        dispatch(command,s,emission_aggregate,0);
        dispatch(command,s,direct,0);
        for(std::uint32_t i=0;i<iterations;i++)
        {
            dispatch(command,s,aggregate,i&1u);
            dispatch(command,s,gather,i&1u);
        }
    }

    void solver::record_refine(vk::CommandBuffer command,std::uint32_t iterations)
    {
        if(!state_ || !command || iterations==0 || (iterations&1u) ||
           iterations>256)
            throw std::invalid_argument("invalid refinement iteration count");
        const auto& s=*state_;
        for(std::uint32_t i=0;i<iterations;i++)
        {
            dispatch(command,s,aggregate,i&1u);
            dispatch(command,s,gather,i&1u);
        }
    }

    void solver::update_emission(std::uint32_t index,vec4 emission)
    {
        if(!state_ || index>=state_->surfaces.size())
            throw std::out_of_range("hierarchy surface index");
        for(const float x:{emission.x,emission.y,emission.z})
            if(!std::isfinite(x) || x<0.f)
                throw std::invalid_argument("invalid emission");
        state_->surfaces[index].value.emission=emission;
        vk::buffer::write(state_->patches,&state_->surfaces[index].value,
                          sizeof(transport::patch),index*sizeof(transport::patch));
    }

    void solver::record_delta(vk::CommandBuffer command,std::uint32_t bounces)
    {
        if(!state_ || !command || bounces==0 || bounces>256)
            throw std::invalid_argument("invalid delta solve");
        const auto& s=*state_;
        dispatch(command,s,inject,3); // Output delta[0], update radiance[0].
        for(std::uint32_t i=0;i<bounces;i++)
        {
            dispatch(command,s,aggregate,2+(i&1u));
            dispatch(command,s,propagate,2+(i&1u));
        }
        dispatch(command,s,emission_aggregate,0);
        dispatch(command,s,direct,0);
    }

    void solver::record_residual(vk::CommandBuffer command)
    {
        if(!state_ || !command)throw std::invalid_argument("invalid residual query");
        const auto& s=*state_;
        dispatch(command,s,aggregate,0);
        command.fillBuffer(s.residual_value.handle,0,4,0u);
        vk::MemoryBarrier transfer{};
        transfer.srcAccessMask=vk::AccessFlagBits::eTransferWrite;
        transfer.dstAccessMask=vk::AccessFlagBits::eShaderRead |
                               vk::AccessFlagBits::eShaderWrite;
        command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                vk::PipelineStageFlagBits::eComputeShader,
                                {},{transfer},{},{});
        dispatch(command,s,residual_kernel,0);
        vk::MemoryBarrier copy{};
        copy.srcAccessMask=vk::AccessFlagBits::eShaderWrite;
        copy.dstAccessMask=vk::AccessFlagBits::eTransferRead;
        command.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                vk::PipelineStageFlagBits::eTransfer,
                                {},{copy},{},{});
        vk::BufferCopy region{};
        region.srcOffset=0;
        region.dstOffset=0;
        region.size=4;
        command.copyBuffer(s.residual_value.handle,s.readback.handle,{region});
    }

    float solver::residual() const
    {
        if(!state_)throw std::logic_error("hierarchy not initialized");
        auto& s=*state_;
        const auto p=vk::buffer::map(s.readback);
        std::uint32_t bits=0;
        std::memcpy(&bits,p,sizeof(bits));
        vk::buffer::unmap(s.readback);
        float value=0.f;
        std::memcpy(&value,&bits,sizeof(value));
         s.info.residual=value;
        const float contraction=std::max(1e-6f,1.f-s.info.operator_norm_upper);
        s.info.discrete_error_bound=value/contraction;
        return value;
    }

    float solver::error_bound() const
    {
        if(!state_)throw std::logic_error("hierarchy not initialized");
        return state_->info.discrete_error_bound;
    }

    const statistics& solver::metrics() const
    {
        if(!state_)throw std::logic_error("hierarchy not initialized");
        return state_->info;
    }

    vk::DescriptorBufferInfo solver::patches_descriptor() const
    {
        if(!state_)throw std::logic_error("hierarchy not initialized");
        return vk::buffer::descriptor(state_->patches);
    }
    vk::DescriptorBufferInfo solver::radiance_descriptor() const
    {
        if(!state_)throw std::logic_error("hierarchy not initialized");
        return vk::buffer::descriptor(state_->radiance[0]);
    }
    vk::DescriptorBufferInfo solver::direct_descriptor() const
    {
        if(!state_)throw std::logic_error("hierarchy not initialized");
        return vk::buffer::descriptor(state_->first);
    }
    vk::DescriptorBufferInfo solver::oriented_descriptor() const
    {
        if(!state_)throw std::logic_error("hierarchy not initialized");
        return vk::buffer::descriptor(state_->oriented);
    }

    void solver::shutdown()
    {
        if(!state_)return;
        auto& s=*state_;
        if(vk::device::get()->handle)
        {
            const auto dev=vk::device::get()->handle;
            for(auto p:s.pipelines)if(p)vk::pipeline::destroy(p);
            if(s.pipeline_layout)vk::pipeline_layout::destroy(s.pipeline_layout);
            if(s.pool)dev.destroyDescriptorPool(s.pool);
            if(s.descriptor_layout)vk::descriptor_layout::destroy(s.descriptor_layout);
            for(auto* b:{&s.patches,&s.nodes,&s.order,&s.offsets,&s.links,
                         &s.history,&s.oriented,&s.readback,&s.radiance[0],
                         &s.radiance[1],&s.delta[0],&s.delta[1],
                         &s.node_values,&s.first,&s.residual_value})
                if(b->handle)vk::buffer::destroy(*b);
        }
        state_.reset();
    }
}
