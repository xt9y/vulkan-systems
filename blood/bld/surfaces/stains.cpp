#include "stains.hpp"
#include "../material/blood.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace bld::surface
{
    namespace
    {
        state *data=nullptr;
        constexpr float pi=3.14159265358979323846f;

        mat4 inverse_affine(const mat4 &m)
        {
            using namespace vk::util::math;
            vec3 a{m(0,0),m(1,0),m(2,0)};
            vec3 b{m(0,1),m(1,1),m(2,1)};
            vec3 c{m(0,2),m(1,2),m(2,2)};
            float det=dot(a,cross(b,c));
            if(std::abs(det)<1e-10f)return mat4::identity();
            vec3 x=cross(b,c)/det,y=cross(c,a)/det,z=cross(a,b)/det;
            mat4 out=mat4::identity();
            out(0,0)=x.x;out(0,1)=x.y;out(0,2)=x.z;
            out(1,0)=y.x;out(1,1)=y.y;out(1,2)=y.z;
            out(2,0)=z.x;out(2,1)=z.y;out(2,2)=z.z;
            const vec3 p{m(0,3),m(1,3),m(2,3)};
            out(0,3)=-dot(x,p);
            out(1,3)=-dot(y,p);
            out(2,3)=-dot(z,p);
            return out;
        }

        float noise(float x)
        {
            return std::sin(x*7.19f)*.085f+
                   std::sin(x*13.67f+1.8f)*.042f+
                   std::cos(x*23.12f)*.02f;
        }

        void rasterize(atlas &image, substrate kind, bool bake=false)
        {
            auto &pixels=bake?image.baked_pixels:image.pixels;
            if(bake)
            {
                if(pixels.size()!=image.pixels.size())
                    pixels.assign(image.pixels.size(),0u);
            }
            else if(image.baked_pixels.size()==image.pixels.size())
                pixels=image.baked_pixels;
            else
                std::fill(pixels.begin(),pixels.end(),0u);

            const int w=int(image.resolution);
            const auto physical=material::surface(kind);
            const std::size_t count=bake?
                std::min<std::size_t>(1,image.marks.size()):
                image.marks.size();
            for(std::size_t i=0;i<count;++i)
            {
                const auto &mark=image.marks[i];
                // Textile wicking spreads over seconds; hard surfaces stay
                // at their contact footprint. No screen-space UV projection.
                const float wick=physical.absorption *
                    (1.f-std::exp(-std::max(0.f,mark.age)*.11f));
                // Porous cloth expands anisotropically along yarn axes.
                // Radius is expressed in actual mesh UV coordinates,
                // calibrated from the impacted triangle's metre/UV metric.
                const float ux=std::clamp(
                    mark.radius_uv.x*physical.spread*(1.f+wick*1.35f),
                    .0005f,.25f);
                const float uy=std::clamp(
                    mark.radius_uv.y*physical.spread*(1.f+wick*.75f),
                    .0005f,.25f);
                const float cx=mark.uv.x*float(w-1);
                const float cy=mark.uv.y*float(w-1);
                const float rx=std::max(1.f,ux*float(w));
                const float ry=std::max(1.f,uy*float(w));
                const auto target=data->targets.find(image.target);
                const vec2 metric=target!=data->targets.end() ?
                    target->second.metres_per_uv : vec2{1.f,1.f};
                const float area=pi*ux*metric.x*uy*metric.y;
                const float film_thickness=mark.volume_ml*.000001f /
                    std::max(area,1e-8f);
                const int reach=int(std::ceil(std::max(rx,ry)*1.5f));
                const float wet=std::clamp(
                    mark.wetness*std::exp(-mark.age/
                        (42.f*physical.drying_scale)),0.f,1.f);
                const int x0=std::max(0,int(cx)-reach),x1=std::min(w-1,int(cx)+reach);
                const int y0=std::max(0,int(cy)-reach),y1=std::min(w-1,int(cy)+reach);
                for(int y=y0;y<=y1;++y)
                    for(int x=x0;x<=x1;++x)
                    {
                        const float dx=(float(x)+.5f-cx)/rx;
                        const float dy=(float(y)+.5f-cy)/ry;
                        const float angle=std::atan2(dy,dx);
                        const float edge=.88f+
                            noise(angle+mark.angle+float(mark.seed%31u)) *
                                (kind==substrate::textile?.55f:1.f);
                        const float radius2=std::sqrt(dx*dx+dy*dy);
                        const float coverage=std::clamp(
                            (edge-radius2)*7.f,0.f,1.f);
                        if(coverage<=0.f)continue;
                        const auto value=std::size_t(y)*w+x;
                        const std::uint32_t old=pixels[value];
                        const float old_cov=float(old>>24u)/255.f;
                        const float new_cov=coverage+old_cov*(1.f-coverage);
                        // Normalized film thickness, where 1 means 1 mm.
                        const float amount=std::clamp(
                            film_thickness/.001f,.008f,1.f);
                        const auto old_depth=float((old>>8u)&255u)/255.f;
                        const auto new_depth=std::max(old_depth,coverage*amount);
                        const auto old_wet=float((old>>16u)&255u)/255.f;
                        const auto new_wet=std::max(old_wet,coverage*wet);
                        const auto dry=std::clamp(physical.absorption,0.f,1.f);
                        const std::uint32_t c=std::uint32_t(new_cov*255.f+.5f);
                        const std::uint32_t v=std::uint32_t(new_wet*255.f+.5f);
                        const std::uint32_t z=std::uint32_t(new_depth*255.f+.5f);
                        const std::uint32_t a=std::uint32_t(dry*255.f+.5f);
                        pixels[value]=(c<<24u)|(v<<16u)|(z<<8u)|a;
                    }
            }
            image.generation=++data->generation;
        }
    }

    state *get() { return data; }

    void init(const options &settings)
    {
        if(data)throw std::logic_error("Blood surfaces initialized twice");
        if(settings.wound_resolution<32 ||
           settings.wound_resolution>2048 ||
           !settings.max_marks)
            throw std::invalid_argument("Invalid blood surface atlas options");
        data=new state{};
        data->settings=settings;
    }

    void register_target(instance_id id,substrate kind,mat4 transform,
                         bool skinned,vec2 metres_per_uv)
    {
        if(!data || !id)return;
        metres_per_uv.x=std::max(.001f,metres_per_uv.x);
        metres_per_uv.y=std::max(.001f,metres_per_uv.y);
        data->targets[id]={kind,transform,skinned,metres_per_uv};
    }

    void set_transform(instance_id id,mat4 transform)
    {
        if(!data || !id)return;
        const auto it=data->targets.find(id);
        if(it!=data->targets.end())it->second.transform=transform;
    }

    void remove_target(instance_id id)
    {
        if(!data)return;
        data->targets.erase(id);
        data->atlases.erase(id);
        auto &stains=data->stains;
        stains.erase(std::remove_if(stains.begin(),stains.end(),
            [id](const rigid_stain &s){return s.target==id;}),stains.end());
    }

    void deposit(instance_id id,vec3 position,vec3 normal,vec2 uv,
                 bool uv_valid,float volume_ml,float velocity,std::uint32_t seed)
    {
        if(!data || !id || !std::isfinite(volume_ml) || volume_ml<=0.f ||
           !std::isfinite(position.x) || !std::isfinite(position.y) ||
           !std::isfinite(position.z) ||
           (uv_valid && (!std::isfinite(uv.x) || !std::isfinite(uv.y) ||
                        uv.x<0.f || uv.y<0.f || uv.x>1.f || uv.y>1.f)))
            return;
        const auto registered=data->targets.find(id);
        if(registered==data->targets.end())return;
        const auto props=material::surface(registered->second.kind);
        const float diameter=std::clamp(
            std::cbrt(volume_ml*.000001f)*1.24f,.001f,.055f);
        const float weber=fld::rheology::weber(material::physical(),
                                              std::abs(velocity),diameter);
        // A bounded spread factor calibrated against dimensionless We;
        // exact impact lamellae depend on substrate and gas conditions.
        const float spread=std::clamp(.6f+std::pow(weber,.25f)*.20f,
                                      .7f,2.5f);

        if(uv_valid)
        {
            auto &atlas=data->atlases[id];
            atlas.target=id;
            atlas.resolution=data->settings.wound_resolution;
            if(atlas.pixels.empty())
                atlas.pixels.assign(std::size_t(atlas.resolution)*
                                    atlas.resolution,0u);
            if(atlas.marks.size()>=data->settings.max_marks)
            {
                // Flatten the oldest mark into a persistent pigment/wetness
                // layer before reclaiming its procedural storage.
                rasterize(atlas,registered->second.kind,true);
                atlas.marks.erase(atlas.marks.begin());
            }
            mark m{};
            m.uv=uv;
            const auto metric=registered->second.metres_per_uv;
            const float world_radius=diameter*.5f*spread;
            m.radius_uv={
                std::clamp(world_radius/metric.x,.0005f,.12f),
                std::clamp(world_radius/metric.y,.0005f,.12f)
            };
            m.volume_ml=volume_ml;
            m.angle=float(seed%628u)/100.f;
            m.seed=seed;
            atlas.marks.push_back(m);
            rasterize(atlas,registered->second.kind);
            return;
        }

        if(registered->second.skinned) return;
        if(data->stains.size()>=data->settings.max_stains)
            data->stains.erase(data->stains.begin());

        // Merge repeated near-identical marks on the same rigid instance.
        for(auto &stain:data->stains)
        {
            if(stain.target!=id)continue;
            const vec4 center=registered->second.transform *
                             (stain.local*vec4{0,0,0,1});
            const vec3 d=position-vec3{center.x,center.y,center.z};
            if(vk::util::math::length_squared(d)>.012f*.012f)continue;
            stain.volume_ml+=volume_ml;
            stain.radius=std::min(.25f,stain.radius+diameter*.12f);
            stain.wetness=1.f;
            return;
        }
        rigid_stain stain{};
        stain.target=id;
        stain.radius=std::clamp(diameter*spread*props.spread,.004f,.20f);
        stain.volume_ml=volume_ml;
        stain.seed=seed;
        stain.kind=registered->second.kind;
        stain.normal=normal;
        const auto n=fld::math::safe_normal(normal);
        const vec3 helper=std::abs(n.y)<.9f ?
            vec3{0,1,0}:vec3{1,0,0};
        const auto tangent=fld::math::safe_normal(
            vk::util::math::cross(helper,n));
        const auto bitangent=vk::util::math::cross(n,tangent);
        stain.world=fld::mat4::identity();
        stain.world(0,0)=tangent.x;
        stain.world(1,0)=tangent.y;
        stain.world(2,0)=tangent.z;
        stain.world(0,1)=bitangent.x;
        stain.world(1,1)=bitangent.y;
        stain.world(2,1)=bitangent.z;
        stain.world(0,2)=n.x;
        stain.world(1,2)=n.y;
        stain.world(2,2)=n.z;
        stain.world(0,3)=position.x;
        stain.world(1,3)=position.y;
        stain.world(2,3)=position.z;
        stain.local=inverse_affine(registered->second.transform)*stain.world;
        data->stains.push_back(stain);
    }

    void update(float dt)
    {
        if(!data || dt<=0.f)return;
        for(auto &[id,image]:data->atlases)
        {
            bool active=false;
            for(auto &m:image.marks)
            {
                m.age+=dt;
                if(m.age<300.f)active=true;
            }
            const bool baked_wet=std::any_of(
                image.baked_pixels.begin(),image.baked_pixels.end(),
                [](std::uint32_t pixel){return ((pixel>>16u)&255u)!=0u;});
            if(active || baked_wet)
            {
                image.update_accumulator+=dt;
                if(image.update_accumulator>=.20f)
                {
                    const float elapsed=image.update_accumulator;
                    image.update_accumulator=0.f;
                    const auto kind=data->targets.at(id).kind;
                    if(baked_wet)
                    {
                        // Pigment coverage is permanent, but excess surface
                        // wetness dries even after a mark has been flattened.
                        const auto props=material::surface(kind);
                        const float decay=std::exp(-elapsed/
                            std::max(1.f,42.f*props.drying_scale));
                        for(auto &pixel:image.baked_pixels)
                        {
                            const auto wet=(pixel>>16u)&255u;
                            const auto remaining=std::uint32_t(
                                std::round(float(wet)*decay));
                            pixel=(pixel&0xff00ffffu)|(remaining<<16u);
                        }
                    }
                    rasterize(image,kind);
                }
            }
        }
        for(auto &mark:data->stains)
        {
            mark.age+=dt;
            const auto props=material::surface(mark.kind);
            mark.wetness=std::exp(-mark.age/
                std::max(1.f,data->settings.drying_time*props.drying_scale));
        }
    }

    const atlas *find(instance_id id)
    {
        if(!data)return nullptr;
        const auto it=data->atlases.find(id);
        return it==data->atlases.end()?nullptr:&it->second;
    }

    void clear()
    {
        if(!data)return;
        data->atlases.clear();
        data->stains.clear();
    }

    void shutdown()
    {
        delete data;
        data=nullptr;
    }
}
