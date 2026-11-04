#include "stains.hpp"
#include "../material/blood.hpp"
#include <fld/material/rheology.hpp>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace bld::surface
{
    namespace
    {
        state *data=nullptr;

        mat4 inverse_affine(const mat4 &m)
        {
            using namespace vk::util::math;
            vec3 a{m(0,0),m(1,0),m(2,0)};
            vec3 b{m(0,1),m(1,1),m(2,1)};
            vec3 c{m(0,2),m(1,2),m(2,2)};
            const float det=dot(a,cross(b,c));
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

        // Fixed point (16 bits per field) in metres, not a procedural
        // opacity stamp. Packing is only for GPU upload; CPU conserves full
        // precision m^3 in each mobile and porous reservoir.
        std::uint32_t encode(const fld::surface::cell &c,float cell_area)
        {
            const auto mobile=std::uint32_t(std::clamp(
                c.mobile/cell_area/.008f,0.f,1.f)*65535.f+.5f);
            const auto retained=std::uint32_t(std::clamp(
                (c.stored+c.residue)/cell_area/.004f,0.f,1.f)*65535.f+.5f);
            return (mobile<<16u)|retained;
        }

        void pack(fld::surface::grid &film,std::vector<std::uint32_t> &pixels)
        {
            const auto area=fld::surface::cell_area(film);
            pixels.resize(film.cells.size());
            for(std::size_t i=0;i<pixels.size();++i)
                pixels[i]=encode(film.cells[i],area);
        }

        fld::porous::properties porous_properties(substrate kind)
        {
            switch(kind)
            {
                case substrate::skin:
                    return {.000035f,.000010f,.00000006f,.00000006f};
                case substrate::textile:
                    return {.0016f,.00030f,.0000016f,.00000045f};
                case substrate::porous:
                    return {.0012f,.00018f,.0000011f,.0000007f};
                case substrate::painted:
                    return {.000005f,.0000008f,0.f,0.f};
                case substrate::metal:
                case substrate::glass:
                    return {0.f,0.f,0.f,0.f};
            }
            return {};
        }

        vec3 unit(vec3 p,vec3 fallback)
        {
            const float length=vk::util::math::length(p);
            return length>1e-8f?p/length:fallback;
        }

        void drip(fld::surface::grid &film,
                  const release_callback &release)
        {
            if(!film.open_edges || !release)return;
            const auto material=bld::material::physical();
            const float length=std::sqrt(material.surface_tension/
                std::max(.001f,material.density*9.81f));
            const float area=fld::surface::cell_area(film);
            // A supported chart declares real open edges. UV seams are NOT
            // physical edges, so never detach there unless the host opts in.
            for(std::uint32_t y=0;y<film.height;++y)
                for(std::uint32_t x=0;x<film.width;++x)
                {
                    if(x!=0 && y!=0 && x+1!=film.width && y+1!=film.height)
                        continue;
                    const auto &c=film.cells[std::size_t(y)*film.width+x];
                    const float thickness=c.mobile/area;
                    if(thickness<length*1.4f)continue;
                    const float volume=std::min(c.mobile*.28f,2.e-7f);
                    if(volume<1.e-9f)continue;
                    fld::emission e{};
                    e.position=fld::surface::world_position(film,x,y);
                    e.velocity={0,-.18f,0};
                    e.volume=volume;
                    e.lifetime=5.f;
                    // Preserve liquid in the film on budget exhaustion.
                    if(release(e))
                        fld::surface::take(film,x,y,volume);
                }
        }

        void accumulate(rigid_stain &stain,vec3 world,float volume,float radius)
        {
            const auto p=inverse_affine(stain.world)*
                         vec4{world.x,world.y,world.z,1.f};
            const auto size=stain.film.metres_per_uv;
            const vec2 uv{
                std::clamp(.5f+p.x/size.x,0.f,1.f),
                std::clamp(.5f+p.y/size.y,0.f,1.f)};
            fld::surface::deposit(stain.film,uv,radius,volume);
            pack(stain.film,stain.pixels);
        }
    }

    state *get() { return data; }

    void init(const options &settings)
    {
        if(data)throw std::logic_error("Blood surfaces initialized twice");
        if(settings.wound_resolution<32 || settings.wound_resolution>2048 ||
           !settings.max_marks || !settings.max_stains)
            throw std::invalid_argument("Invalid blood surface options");
        data=new state{};
        data->settings=settings;
    }

    void register_target(instance_id id,substrate kind,mat4 transform,
                         bool skinned,vec2 metric)
    {
        if(!data || !id)return;
        metric.x=std::max(.001f,metric.x);
        metric.y=std::max(.001f,metric.y);
        auto found=data->targets.find(id);
        if(found!=data->targets.end())
        {
            found->second.transform=transform;
            // Do not reset the film metric/chart on every droplet contact.
            // Otherwise its numerical cell area changes and loses mass.
            return;
        }
        data->targets[id]={kind,transform,skinned,metric};
    }

    void set_transform(instance_id id,mat4 transform)
    {
        if(!data)return;
        const auto target=data->targets.find(id);
        if(target==data->targets.end())return;
        target->second.transform=transform;
        const auto atlas=data->atlases.find(id);
        if(atlas!=data->atlases.end())
            atlas->second.film.transform=transform;
        for(auto &stain:data->stains)
            if(stain.target==id)
            {
                stain.world=transform*stain.local;
                stain.film.transform=stain.world;
            }
        ++data->generation;
    }

    void register_geometry(instance_id id,
        const std::vector<fld::surface::chart_vertex>& vertices,
        const std::vector<std::uint32_t>& indices,
        bool physical_chart_edges)
    {
        if(!data || !data->targets.count(id))return;
        auto &image=data->atlases[id];
        image.target=id;image.resolution=data->settings.wound_resolution;
        if(image.film.cells.empty())
            fld::surface::init(image.film,image.resolution,image.resolution,
                data->targets.at(id).metres_per_uv);
        image.film.transform=data->targets.at(id).transform;
        fld::surface::set_chart(image.film,vertices,indices);
        // Only explicitly supplied physical edges may shed a droplet.
        // UV seams are not an acceptable detachment condition.
        image.film.open_edges=physical_chart_edges;
        pack(image.film,image.pixels);
        image.generation=++data->generation;
    }

    void remove_target(instance_id id)
    {
        if(!data)return;
        data->targets.erase(id);
        data->atlases.erase(id);
        data->rigid_targets.erase(id);
        auto &stains=data->stains;
        stains.erase(std::remove_if(stains.begin(),stains.end(),
            [id](const rigid_stain &s){return s.target==id;}),stains.end());
        ++data->generation;
    }

    void deposit(instance_id id,vec3 position,vec3 normal,vec2 uv,
                 bool uv_valid,float volume_ml,float velocity,
                 std::uint32_t seed)
    {
        if(!data || !id || !std::isfinite(volume_ml) || volume_ml<=0.f ||
           !std::isfinite(position.x) || !std::isfinite(position.y) ||
           !std::isfinite(position.z))return;
        const auto it=data->targets.find(id);
        if(it==data->targets.end())return;
        const auto props=material::surface(it->second.kind);
        const float volume=volume_ml*1.e-6f;
        const float diameter=std::clamp(std::cbrt(volume)*1.24f,
                                        .001f,.055f);
        const float weber=fld::rheology::weber(material::physical(),
                                               std::abs(velocity),diameter);
        const float spread=std::clamp(
            .6f+std::pow(std::max(0.f,weber),.25f)*.20f,.7f,2.5f);
        const float radius=std::clamp(diameter*.5f*spread,.002f,.07f);

        const bool atlas_surface=it->second.skinned ||
            it->second.kind==substrate::textile ||
            it->second.kind==substrate::skin;
        if(uv_valid && atlas_surface &&
           std::isfinite(uv.x) && std::isfinite(uv.y))
        {
            uv.x=std::clamp(uv.x,0.f,1.f);
            uv.y=std::clamp(uv.y,0.f,1.f);
            auto &image=data->atlases[id];
            image.target=id;image.resolution=data->settings.wound_resolution;
            if(image.film.cells.empty())
            {
                fld::surface::init(image.film,image.resolution,image.resolution,
                                   it->second.metres_per_uv);
                fld::surface::set_frame(image.film,normal,it->second.transform);
            }
            // The marks vector is capped impact history only; it no longer
            // rasterizes permanent synthetic circles over physical liquid.
            if(image.marks.size()>=data->settings.max_marks)
                image.marks.erase(image.marks.begin());
            mark record{};
            record.uv=uv;
            record.volume_ml=volume_ml;
            record.seed=seed;
            image.marks.push_back(record);
            fld::surface::deposit(image.film,uv,radius,volume);
            pack(image.film,image.pixels);
            image.generation=++data->generation;
            return;
        }

        // Free-film patches attached to world surfaces. Their texture
        // changes through the same PDE as the skin film, with NO UV stretching
        // over an entire wall/floor.
        if(it->second.skinned)return;
        for(auto &stain:data->stains)
        {
            if(stain.target!=id)continue;
            const auto center=stain.world*vec4{0,0,0,1};
            const vec3 delta=position-vec3{center.x,center.y,center.z};
            if(vk::util::math::length_squared(delta)>
               stain.radius*stain.radius*2.f)continue;
            accumulate(stain,position,volume,radius);
            stain.volume_ml+=volume_ml;
            stain.wetness=1.f;
            ++data->generation;
            return;
        }

        if(data->stains.size()>=data->settings.max_stains)
        {
            const auto removed=data->stains.front().target;
            data->stains.erase(data->stains.begin());
            if(std::none_of(data->stains.begin(),data->stains.end(),
                 [removed](const rigid_stain &s){return s.target==removed;}))
                data->rigid_targets.erase(removed);
        }
        rigid_stain stain{};
        stain.target=id;
        stain.radius=std::clamp(radius*props.spread*1.5f,.012f,.18f);
        stain.volume_ml=volume_ml;
        stain.seed=seed;
        stain.kind=it->second.kind;
        stain.normal=unit(normal,{0,1,0});
        const vec3 helper=std::abs(stain.normal.y)<.9f ?
            vec3{0,1,0}:vec3{1,0,0};
        const auto u=unit(vk::util::math::cross(helper,stain.normal),{1,0,0});
        const auto v=vk::util::math::cross(stain.normal,u);
        stain.world=mat4::identity();
        stain.world(0,0)=u.x;stain.world(1,0)=u.y;stain.world(2,0)=u.z;
        stain.world(0,1)=v.x;stain.world(1,1)=v.y;stain.world(2,1)=v.z;
        stain.world(0,2)=stain.normal.x;
        stain.world(1,2)=stain.normal.y;
        stain.world(2,2)=stain.normal.z;
        stain.world(0,3)=position.x;
        stain.world(1,3)=position.y;
        stain.world(2,3)=position.z;
        stain.local=inverse_affine(it->second.transform)*stain.world;
        // Small independent patches: each runs its own conservative film
        // without allocating one enormous atlas for the whole ground.
        constexpr std::uint32_t resolution=48;
        fld::surface::init(stain.film,resolution,resolution,
                           {stain.radius*4.f,stain.radius*4.f});
        fld::surface::set_frame(stain.film,{0,0,1},stain.world);
        accumulate(stain,position,volume,radius);
        data->stains.push_back(std::move(stain));
        data->rigid_targets.insert(id);
        ++data->generation;
    }

    void update(float dt,const release_callback &release)
    {
        if(!data || !std::isfinite(dt) || dt<=0.f)return;
        const auto fluid=material::physical();
        // A 30 Hz surface-film step is sufficient for millimetre-scale
        // blood films and halves CPU work on a 60 Hz renderer. The fixed-
        // volume solver still substeps for stability and does not tie
        // transport to GPU frame rate.
        for(auto &[id,image]:data->atlases)
        {
            const auto target=data->targets.find(id);
            if(target==data->targets.end() || image.film.cells.empty())continue;
            image.update_accumulator+=dt;
            if(image.update_accumulator>=1.f/30.f)
            {
                const auto elapsed=std::min(.1f,image.update_accumulator);
                image.update_accumulator=0.f;
                fld::surface::update(image.film,elapsed,fluid,
                                      porous_properties(target->second.kind));
                drip(image.film,release);
                pack(image.film,image.pixels);
                image.generation=++data->generation;
            }
            for(auto &m:image.marks)m.age+=dt;
        }
        for(auto &stain:data->stains)
        {
            stain.age+=dt;
            const auto properties=material::surface(stain.kind);
            stain.wetness=std::exp(-stain.age/
                std::max(1.f,data->settings.drying_time*
                         properties.drying_scale));
        }
        data->rigid_update_accumulator+=dt;
        if(data->rigid_update_accumulator>=1.f/30.f)
        {
            const auto elapsed=std::min(.1f,data->rigid_update_accumulator);
            data->rigid_update_accumulator=0.f;
            for(auto &stain:data->stains)
            {
                fld::surface::update(stain.film,elapsed,fluid,
                                      porous_properties(stain.kind));
                drip(stain.film,release);
                pack(stain.film,stain.pixels);
            }
            if(!data->stains.empty())++data->generation;
        }
    }

    const atlas *find(instance_id id)
    {
        if(!data)return nullptr;
        const auto it=data->atlases.find(id);
        return it==data->atlases.end()?nullptr:&it->second;
    }

    double volume()
    {
        if(!data)return 0.;
        double result=0.;
        for(const auto &[id,image]:data->atlases)
            result+=fld::surface::volume(image.film);
        for(const auto &stain:data->stains)
            result+=fld::surface::volume(stain.film);
        return result;
    }

    void clear()
    {
        if(!data)return;
        data->atlases.clear();
        data->stains.clear();
        data->rigid_targets.clear();
        ++data->generation;
    }

    void shutdown()
    {
        delete data;
        data=nullptr;
    }
}
