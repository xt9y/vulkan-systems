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
        const float velocity=std::clamp(event.velocity,0.f,25.f);

        fld::splash_source source{};
        source.position=event.position;
        source.normal=outward;
        source.velocity=outward*velocity + travel*(velocity*.18f);
        source.volume=std::clamp(event.amount_ml,.1f,30.f)*.000001f;
        source.group=std::uint32_t(event.id ^ (event.id>>32u));
        source.phase=data->phase;
        source.seed=event.seed;
        source.lifetime=4.f;
        if(const auto *surfaces=surface::get())
        {
            const auto owner=surfaces->targets.find(event.target);
            if(owner!=surfaces->targets.end())
                source.surface=material::interaction(owner->second.kind);
        }
        // Both blood and other liquids use EXACTLY this generic solver path.
        const auto splash=fld::system::splash(source);
        if(event.target && splash.retained_volume>0.f)
            surface::deposit(event.target,event.position,event.normal,
                             event.uv,event.uv_valid,
                             splash.retained_volume*1e6f,
                             velocity,event.seed);
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
                // Unregistered scene geometry is a passive hard wall.
                // Never delete liquid mass because a target has no stain
                // handler. The host can register a surface at any time.
                if(!surfaces || !surfaces->targets.count(hit.target))
                {
                    hit.material_response=false;
                    hit.absorb=false;
                    hit.deposition_fraction=0.f;
                    hit.restitution=.02f;
                    hit.friction=.65f;
                    return hit;
                }
                const auto it=surfaces->targets.find(hit.target);
                // The blood layer only describes the receiving surface.
                // Actual wetting, absorption and rebound are solved in fld.
                hit.surface=material::interaction(it->second.kind);
                hit.material_response=true;
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
