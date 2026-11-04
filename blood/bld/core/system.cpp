#include "system.hpp"
#include "../math/geometry.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace bld::system
{
    namespace { state *data = nullptr; }

    state *get() { return data; }

    void init(const settings &options)
    {
        if (data) throw std::logic_error("Blood system already initialized");
        if (!(options.fixed_step > 0.f) ||
            options.fixed_step > .05f ||
            options.wound_resolution < 16 ||
            options.wound_resolution > 2048 ||
            (options.wound_resolution % 2u) != 0u)
            throw std::invalid_argument("Invalid blood system settings");
        data = new state{};
        data->options = options;
        try
        {
            surfaces::init(options);
            wounds::init(options);
            simulation::init(options);
        }
        catch (...)
        {
            simulation::shutdown();
            wounds::shutdown();
            surfaces::shutdown();
            delete data;
            data = nullptr;
            throw;
        }
    }

    impact hit(impact event)
    {
        if (!data) throw std::logic_error("Initialize blood before impact");
        if (!event.id) event.id = data->sequence++;
        if (!event.seed) event.seed = std::uint32_t(event.id) ^
                                           std::uint32_t(event.id >> 32);
        apply(event);
        return event;
    }

    void apply(const impact &event)
    {
        if (!data || event.type != material::flesh) return;
        if (!std::isfinite(event.position.x) || !std::isfinite(event.position.y) ||
            !std::isfinite(event.position.z) ||
            !std::isfinite(event.normal.x) || !std::isfinite(event.normal.y) ||
            !std::isfinite(event.normal.z) ||
            !std::isfinite(event.direction.x) || !std::isfinite(event.direction.y) ||
            !std::isfinite(event.direction.z) ||
            !std::isfinite(event.energy) || event.energy <= 0.f) return;

        simulation::emit(event);
        if (event.uv_valid && event.target)
        {
            auto seed = event.seed;
            wounds::stamp(event.target, event.uv,
                          .023f + math::unit(seed) * .027f,
                          .85f + math::unit(seed)*.15f, seed);
            const auto attached = surfaces::get()->transforms.find(event.target);
            if (attached != surfaces::get()->transforms.end())
            {
                const auto local = math::transform_point(
                    math::inverse_affine(attached->second), event.position);
                wounds::source_at(event.target, local, event.energy, seed);
            }
        }
        else if (event.target)
            surfaces::add(event.target, event.position, event.normal,
                          .24f, event.seed, false, 1.2f);

        if (data->pending_backdrops.size() >= 32)
            data->pending_backdrops.erase(data->pending_backdrops.begin());
        data->pending_backdrops.push_back(event);
    }

    void update(float dt, const collision_callback &world)
    {
        if (!data) return;
        const float bounded = std::max(0.f, std::min(dt, .12f));
        if (bounded == 0) return;
        data->accumulator += bounded;
        const double step = data->options.fixed_step;
        for (int iterations = 0;
             data->accumulator >= step && iterations < 24;
             ++iterations)
        {
            if (world && !data->pending_backdrops.empty())
            {
                for (const auto &event : data->pending_backdrops)
                {
                    auto travel = math::direction(event.direction, {0, 0, 1});
                    auto a = event.position + travel * .07f;
                    for (int i = 0; i < 5; ++i)
                    {
                        const auto b = a + travel * 4.f;
                        const auto struck = world(a, b);
                        if (!struck.hit) break;
                        if (struck.target != event.target)
                        {
                            auto seed = event.seed;
                            surfaces::add(struck, .33f + .21f * math::unit(seed),
                                          seed, false, 1.45f);
                            break;
                        }
                        a = struck.position + travel * .045f;
                    }
                }
                data->pending_backdrops.clear();
            }
            simulation::update(float(step), world);
            for (auto p : wounds::update(float(step)))
                simulation::drip(p, std::uint32_t(++data->sequence));
            surfaces::update(float(step));
            ++data->steps;
            data->accumulator -= step;
        }
        if (data->accumulator > step*2) data->accumulator = step;
        if (!world) data->pending_backdrops.clear();
    }

    void clear()
    {
        if (!data) return;
        const auto options = data->options;
        simulation::shutdown();
        wounds::shutdown();
        surfaces::shutdown();
        surfaces::init(options);
        wounds::init(options);
        simulation::init(options);
        data->pending_backdrops.clear();
        data->accumulator = 0;
    }

    void shutdown()
    {
        simulation::shutdown();
        wounds::shutdown();
        surfaces::shutdown();
        delete data;
        data = nullptr;
    }
}
