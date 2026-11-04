#include "system.hpp"
#include "../simulation/solver.hpp"
#include "../math/kernels.hpp"
#include "../material/contact.hpp"
#include "../material/rheology.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace fld::system
{
    namespace
    {
        state *data = nullptr;
        constexpr float pi = 3.14159265358979323846f;
    }

    state *get() { return data; }

    void init(const options &options)
    {
        if (data) throw std::logic_error("Fluid system already initialized");
        if (!std::isfinite(options.spacing) || options.spacing <= 0 ||
            !std::isfinite(options.smoothing_radius) ||
            options.smoothing_radius < 1.5f*options.spacing ||
            !std::isfinite(options.fixed_step) ||
            options.fixed_step <= 0 || options.fixed_step > .05f ||
            !options.pressure_iterations || options.pressure_iterations > 16 ||
            !options.max_substeps || options.max_substeps > 32)
            throw std::invalid_argument("Invalid fluid solver options");
        data = new state{};
        data->settings = options;
        data->materials.push_back(material{});
        data->particles.reserve(options.max_particles);
    }

    std::uint32_t add_material(const fld::material &m)
    {
        if (!data || !std::isfinite(m.density) || m.density <= 0.f ||
            !std::isfinite(m.viscosity_infinite) ||
            m.viscosity_infinite <= 0.f ||
            !std::isfinite(m.viscosity_zero) ||
            m.viscosity_zero < m.viscosity_infinite ||
            !std::isfinite(m.surface_tension) ||
            m.surface_tension <= 0.f ||
            !std::isfinite(m.flow_index) ||
            m.flow_index <= 0.f || m.flow_index > 2.f ||
            !std::isfinite(m.yasuda_exponent) ||
            m.yasuda_exponent < .1f || m.yasuda_exponent > 8.f ||
            !std::isfinite(m.relaxation) || m.relaxation < 0.f)
            throw std::invalid_argument("Invalid fluid material or state");
        data->materials.push_back(m);
        return std::uint32_t(data->materials.size()-1);
    }

    std::uint32_t emit(const fld::emission &source)
    {
        if(!data || source.phase>=data->materials.size() ||
           !std::isfinite(source.volume) || source.volume<=0.f ||
           !std::isfinite(source.position.x) ||
           !std::isfinite(source.position.y) ||
           !std::isfinite(source.position.z) ||
           !std::isfinite(source.velocity.x) ||
           !std::isfinite(source.velocity.y) ||
           !std::isfinite(source.velocity.z) ||
           data->particles.size()>=data->settings.max_particles)
            return 0;

        const auto &settings=data->settings;
        const float space=settings.spacing;
        const float sphere=std::cbrt(source.volume*3.f/(4.f*pi));
        const int reach=std::max(1,int(std::ceil(sphere/space)));
        const std::size_t limit=settings.max_particles-data->particles.size();
        if(reach>int(std::cbrt(double(settings.max_particles))*4.+8.))
            return 0;
        std::vector<vec3> positions;
        positions.reserve(std::min<std::size_t>(limit,512));
        for(int z=-reach;z<=reach;++z)
            for(int y=-reach;y<=reach;++y)
                for(int x=-reach;x<=reach;++x)
                {
                    const vec3 v{float(x)*space,float(y)*space,float(z)*space};
                    if(vk::util::math::length_squared(v)>sphere*sphere)
                        continue;
                    if(positions.size()==limit)
                        return 0;  // No partially represented volume.
                    positions.push_back(v);
                }
        if(positions.empty())positions.push_back({});
        const float volume_each=source.volume/float(positions.size());
        const float density=data->materials[source.phase].density;
        const float radius=std::cbrt(
            volume_each*3.f/(4.f*pi));
        std::uint32_t seed=source.seed ? source.seed : 1;
        for(auto position:positions)
        {
            const vec3 jitter{
                (math::random01(seed)-.5f)*space*.12f,
                (math::random01(seed)-.5f)*space*.12f,
                (math::random01(seed)-.5f)*space*.12f
            };
            const vec3 direction=math::safe_normal(
                position+vec3{0.f,space*.002f,0.f});
            particle p{};
            p.position=source.position+position+jitter;
            p.previous=p.predicted=p.position;
            p.velocity=source.velocity+direction*
                (source.spread*math::random01(seed));
            p.radius=radius;
            p.mass=density*volume_each;
            p.lifetime=std::max(.01f,source.lifetime);
            p.group=source.group;
            p.phase=source.phase;
            data->particles.push_back(p);
        }
        data->stats.emitted_mass_kg+=source.volume*density;
        data->stats.live_mass_kg+=source.volume*density;
        return std::uint32_t(positions.size());
    }

    splash_result splash(const fld::splash_source &source)
    {
        splash_result result{};
        if(!data || source.phase>=data->materials.size() ||
           !std::isfinite(source.volume) || source.volume<=0.f ||
           !std::isfinite(source.velocity.x) ||
           !std::isfinite(source.velocity.y) ||
           !std::isfinite(source.velocity.z) ||
           !std::isfinite(source.position.x) ||
           !std::isfinite(source.position.y) ||
           !std::isfinite(source.position.z))return result;

        const auto &material=data->materials[source.phase];
        const float speed=math::norm(source.velocity);
        const float radius=data->settings.spacing*.5f;
        particle representative{};
        representative.radius=radius;
        representative.velocity=source.velocity;
        representative.mass=material.density *
            std::pow(data->settings.spacing,3.f);
        collision probe{};
        probe.hit=true;
        probe.material_response=true;
        probe.surface=source.surface;
        const auto info=wetting::resolve(material,representative,probe);
        result.weber=info.weber;
        result.reynolds=info.reynolds;
        const float deposited=std::clamp(probe.deposition_fraction,0.f,1.f);
        result.retained_volume=source.volume*deposited;
        const float available=source.volume-result.retained_volume;
        if(available<=0.f)return result;

        const auto axis=math::safe_normal(source.normal);
        const vec3 tangent=std::abs(axis.y)<.94f ?
            math::safe_normal(vk::util::math::cross(axis,vec3{0,1,0})) :
            math::safe_normal(vk::util::math::cross(axis,vec3{1,0,0}));
        const auto bitangent=vk::util::math::cross(axis,tangent);

        // This governs the sheet opening angle and secondary breakup.
        // The solver, not the blood preset, decides from We, Re and wetting.
        const float oh=info.ohnesorge;
        const float breakup=std::clamp(
            info.splash_parameter/(57.7f+info.splash_parameter),0.f,1.f);
        // Viscous splash sheets open more narrowly than ballistic spray.
        // Retain the solver's breakup dependence instead of randomizing
        // each droplet into an unrelated, high-speed trajectory.
        const float opening=std::clamp(
            .10f+breakup*.30f/(1.f+oh*8.f),.10f,.48f);
        emission initial{};
        // Begin the bulk ejecta just outside the impacted surface, including
        // the finite radius of the starting fluid clump. Otherwise the
        // collision callback immediately reabsorbs half the sheet.
        const float source_radius=std::cbrt(
            available*3.f/(4.f*pi));
        initial.position=source.position+axis*(
            source_radius+data->settings.spacing*1.5f);
        initial.velocity=axis*speed;
        initial.spread=0.f;
        initial.volume=available;
        initial.lifetime=source.lifetime;
        initial.phase=source.phase;
        initial.group=source.group;
        initial.seed=source.seed;
        const auto before=data->particles.size();
        const auto count=emit(initial);
        result.particles=count;
        if(!count)
        {
            // Do not silently lose fluid when a tiny impact is below the
            // discretization limit or the particle budget is exhausted.
            result.retained_volume=source.volume;
            return result;
        }
        float emitted=0.f;
        std::uint32_t seed=source.seed ? source.seed : 1;
        const vec3 lateral=source.velocity-axis*
            vk::util::math::dot(source.velocity,axis);
        for(std::size_t i=before;i<data->particles.size();++i)
        {
            auto &p=data->particles[i];
            const float angle=math::random01(seed)*2.f*pi;
            const vec3 radial=tangent*std::cos(angle)+
                              bitangent*std::sin(angle);
            const float eccentricity=math::random01(seed);
            const float theta=opening*std::sqrt(eccentricity);
            // Most impact energy is spent on deformation, wetting and
            // cohesion. Keep the particle neighborhood moving as a sheet;
            // radial breakup remains controlled by the fluid's We/Oh.
            const float forward_speed=speed*
                (.24f+.17f*math::random01(seed))*
                (1.f-source.surface.wetting*.10f);
            const float outward_speed=speed*
                (.34f+.12f*math::random01(seed));
            p.velocity=axis*(std::cos(theta)*forward_speed)+
                       radial*(std::sin(theta)*outward_speed)+
                       lateral*.12f;
            emitted+=p.mass/material.density;
        }
        result.emitted_volume=emitted;
        // Quantization loses a fractional particle volume. Deposit the
        // residual on the receiving surface to preserve mass globally.
        result.retained_volume=std::max(0.f,
            source.volume-result.emitted_volume);
        return result;
    }

    void update(float seconds, const collision_callback &collision,
                const contact_callback &contact)
    {
        if (!data || seconds <= 0 || !std::isfinite(seconds)) return;
        const float step = data->settings.fixed_step;
        data->remainder += std::min(seconds, .1f);
        std::uint32_t iterations = 0;
        while (data->remainder >= step &&
               iterations < data->settings.max_substeps)
        {
            simulation::solve(*data, step, collision, contact);
            data->remainder -= step;
            ++data->stats.steps;
            ++iterations;
        }
        if (data->remainder > step*2) data->remainder = step;
    }

    void clear()
    {
        if (!data) return;
        data->particles.clear();
        data->stats = {};
        data->remainder = 0;
    }

    void clear_phase(std::uint32_t phase)
    {
        if(!data)return;
        auto &ps=data->particles;
        ps.erase(std::remove_if(ps.begin(),ps.end(),
            [phase](const particle &p){return p.phase==phase;}),
            ps.end());
        data->stats.live_mass_kg=0;
        for(const auto &p:ps)data->stats.live_mass_kg+=p.mass;
    }

    void shutdown()
    {
        delete data;
        data = nullptr;
    }
}
