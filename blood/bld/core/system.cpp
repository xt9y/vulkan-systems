#include "system.hpp"
#include "../material/blood.hpp"
#include "../surfaces/stains.hpp"
#include "../network/events.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace bld::system
{
    namespace { state *data=nullptr; }

    state *get() { return data; }

    void init(const options &options)
    {
        if(data)throw std::logic_error("Blood system already initialized");
        data=new state{};
        data->settings=options;
        try
        {
            if(!fld::system::get())
            {
                // Keep a shared solver for other liquid systems.
                fld::system::init(options.fluid);
                data->owns_fluid=true;
            }
            data->phase=fld::system::add_material(material::physical());
            surface::init(options);
        }
        catch(...)
        {
            surface::shutdown();
            if(data->owns_fluid)fld::system::shutdown();
            delete data;
            data=nullptr;
            throw;
        }
    }

    impact hit(impact event)
    {
        if(!data)throw std::logic_error("Blood system not initialized");
        if(!event.id)event.id=data->sequence++;
        if(!event.seed)
            event.seed=std::uint32_t(event.id ^ (event.id>>32u));
        apply(event);
        return event;
    }

    void apply(const impact &event)
    {
        if(!data || !fld::system::get() ||
           !std::isfinite(event.position.x) ||
           !std::isfinite(event.position.y) ||
           !std::isfinite(event.position.z) ||
           !std::isfinite(event.amount_ml) ||
           !std::isfinite(event.velocity) ||
           event.amount_ml<=0 || event.velocity<0)return;

        const auto outward=fld::math::safe_normal(event.normal);
        const auto travel=fld::math::safe_normal(event.direction);
        const float volume=std::clamp(event.amount_ml,.1f,30.f);
        const float velocity=std::clamp(event.velocity,0.f,25.f);
        const float deposit_ml=volume*.18f;
        if(event.target)
        {
            surface::deposit(event.target,event.position,event.normal,
                             event.uv,event.uv_valid,deposit_ml,
                             velocity,event.seed);
        }

        fld::emission plume{};
        plume.position=event.position+outward*.015f;
        plume.velocity=outward*(velocity*.76f)+travel*(velocity*.22f);
        plume.volume=volume*.82f*.000001f;
        plume.spread=.50f+velocity*.15f;
        plume.group=std::uint32_t(event.id ^ (event.id>>32u));
        plume.phase=data->phase;
        plume.seed=event.seed;
        plume.lifetime=4.f;
        fld::system::emit(plume);
        ++data->impacts;
    }

    void update(float dt,const collision_callback &world)
    {
        if(!data)return;
        network::dispatch();
        // Surface class decides whether the impacted portion adheres, wicks,
        // or rebounds. The solver computes liquid motion and contacts.
        fld::system::update(dt,
            [&](fld::vec3 from,fld::vec3 to,float radius)->fld::collision
            {
                if(!world)return {};
                auto hit=world(from,to,radius);
                if(!hit.hit)return hit;
                const auto *surfaces=surface::get();
                const auto it=surfaces?surfaces->targets.find(hit.target):
                                         surface::state{}.targets.end();
                if(!surfaces || it==surfaces->targets.end())
                {
                    hit.absorb=true;
                    return hit;
                }
                const auto kind=it->second.kind;
                if(kind==substrate::skin ||
                   kind==substrate::textile ||
                   kind==substrate::porous)
                {
                    hit.absorb=true;
                }
                else
                {
                    hit.absorb=false;
                    hit.restitution=kind==substrate::glass?.12f:.07f;
                    hit.friction=.40f;
                }
                return hit;
            },
            [&](const fld::particle &p,const fld::collision &hit)
            {
                if(p.phase!=data->phase || !hit.target)return;
                const float speed=fld::math::norm(p.velocity);
                const auto &physical=material::physical();
                const float volume_ml=p.mass/physical.density*1e6f;
                surface::deposit(hit.target,hit.position,hit.normal,
                                 hit.uv,hit.uv_valid,volume_ml,
                                 speed,p.group);
                ++data->deposited;
            });
        surface::update(dt);
    }

    void clear()
    {
        if(!data)return;
        // The shared fluid solver may contain other liquid phases.
        if(data->owns_fluid)fld::system::clear();
        surface::clear();
        network::reset();
        data->impacts=data->deposited=0;
    }

    void shutdown()
    {
        if(!data)return;
        network::shutdown();
        surface::shutdown();
        if(data->owns_fluid)fld::system::shutdown();
        delete data;
        data=nullptr;
    }
}
