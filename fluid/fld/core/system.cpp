#include "system.hpp"
#include "../simulation/solver.hpp"
#include "../math/kernels.hpp"
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
            m.flow_index <= 0.f || m.flow_index > 2.f ||
            m.relaxation < 0.f)
            throw std::invalid_argument("Invalid fluid material or state");
        data->materials.push_back(m);
        return std::uint32_t(data->materials.size()-1);
    }

    std::uint32_t emit(const fld::emission &source)
    {
        if (!data || source.phase >= data->materials.size() ||
            source.volume <= 0.f || !std::isfinite(source.volume))
            return 0;

        const auto &settings = data->settings;
        const auto space = settings.spacing;
        const float sphere = std::cbrt(source.volume * 3.f / (4.f*pi));
        const int reach = std::max(1, int(std::ceil(sphere / space)));
        const float volume = space * space * space;
        const float mass = data->materials[source.phase].density * volume;
        if (data->particles.size() >= settings.max_particles) return 0;
        const auto limit = settings.max_particles -
            std::uint32_t(data->particles.size());
        std::uint32_t written = 0;
        auto seed = source.seed ? source.seed : 1u;

        for (int z = -reach; z <= reach && written < limit; ++z)
            for (int y = -reach; y <= reach && written < limit; ++y)
                for (int x = -reach; x <= reach && written < limit; ++x)
                {
                    const vec3 base{float(x)*space, float(y)*space, float(z)*space};
                    if (vk::util::math::length_squared(base) > sphere*sphere)
                        continue;
                    const vec3 jitter{
                        (math::random01(seed)-.5f)*space*.12f,
                        (math::random01(seed)-.5f)*space*.12f,
                        (math::random01(seed)-.5f)*space*.12f
                    };
                    const vec3 outward = math::safe_normal(
                        base + vec3{0.f, space*.002f, 0.f});
                    particle p{};
                    p.position = source.position + base + jitter;
                    p.previous = p.predicted = p.position;
                    p.velocity = source.velocity + outward *
                        (source.spread * math::random01(seed));
                    p.radius = space*.5f;
                    p.mass = mass;
                    p.lifetime = std::max(.01f, source.lifetime);
                    p.group = source.group;
                    p.phase = source.phase;
                    data->particles.push_back(p);
                    ++written;
                }
        return written;
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

    void shutdown()
    {
        delete data;
        data = nullptr;
    }
}
