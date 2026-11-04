#include "decals.hpp"
#include "../math/geometry.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace bld::surfaces
{
    namespace { state *data = nullptr; }

    state *get() { return data; }

    void init(const settings &options)
    {
        if (data) throw std::logic_error("Blood surfaces already initialized");
        data = new state{};
        data->capacity = std::max(1u, options.max_decals);
        data->max_pools = options.max_pools;
    }

    void set_transform(instance_id id, const mat4 &model)
    {
        if (data && id) data->transforms[id] = model;
    }

    void remove_transform(instance_id id)
    {
        if (!data) return;
        data->transforms.erase(id);
        for (auto &decal : data->decals)
        {
            if (decal.target != id) continue;
            // Keep the last world-space projection if the rigid owner disappears.
            // The last resolved world transform is retained when detached.
            decal.target = 0;
        }
    }

    void add(const collision &hit, float size, std::uint32_t seed,
             bool pool, float stretch)
    {
        if (!hit.hit || hit.skinned) return;
        add(hit.target, hit.position, hit.normal, size, seed, pool, stretch);
    }

    void add(instance_id target, vec3 position, vec3 normal, float size,
             std::uint32_t seed, bool pool, float stretch)
    {
        if (!data || data->capacity == 0) return;
        if (pool)
        {
            const auto count = std::count_if(
                data->decals.begin(), data->decals.end(),
                [](const decal &d) { return d.pool; });
            if (std::uint32_t(count) >= data->max_pools) return;
        }
        if (data->decals.size() >= data->capacity)
            data->decals.erase(data->decals.begin());
        auto n = math::direction(normal);
        auto frame = math::projector(position, n, size, size * stretch,
                                     math::unit(seed) * 6.2831853f);
        if (!pool && stretch > 1.2f)
        {
            // Orient the elongation downhill on non-horizontal surfaces.
            const auto g = vec3{0, -1, 0};
            const auto down = g - n * vk::util::math::dot(g, n);
            if (vk::util::math::length_squared(down) > .15f)
            {
                const auto v = math::direction(down);
                const auto u = math::direction(vk::util::math::cross(v, n));
                for (int i = 0; i < 3; ++i)
                {
                    frame(i, 0) = (&u.x)[i] * size;
                    frame(i, 1) = (&v.x)[i] * size * stretch;
                }
            }
        }
        decal d;
        d.target = target;
        d.world = frame;
        d.radius = pool ? .08f : size;
        d.final_radius = size;
        d.wetness = .85f + math::unit(seed) * .15f;
        d.variant = 1u + math::random(seed) % 4u;
        d.pool = pool;
        d.lifetime = pool ? 360.f : 240.f;
        const auto owner = data->transforms.find(target);
        if (owner != data->transforms.end())
            d.local = math::inverse_affine(owner->second) * frame;
        else d.target = 0;
        data->decals.push_back(d);
    }

    void update(float dt)
    {
        if (!data) return;
        for (auto &d : data->decals)
        {
            d.age += dt;
            if (d.pool)
            {
                const float t = std::min(1.f, d.age / 4.f);
                d.radius = .08f + (d.final_radius - .08f) * t * t * (3 - 2*t);
            }
        }
        auto &v = data->decals;
        v.erase(std::remove_if(v.begin(), v.end(),
                [](const decal &d) { return d.age >= d.lifetime; }), v.end());
    }

    std::vector<projector> resolve()
    {
        std::vector<projector> result;
        if (!data) return result;
        result.reserve(data->decals.size());
        for (const auto &d : data->decals)
        {
            auto frame = d.world;
            const auto found = data->transforms.find(d.target);
            if (d.target && found != data->transforms.end())
                frame = found->second * d.local;
            if (d.pool)
            {
                const float factor = d.radius / std::max(.0001f, d.final_radius);
                for (int i = 0; i < 3; ++i)
                {
                    frame(i, 0) *= factor;
                    frame(i, 1) *= factor;
                }
            }
            const vec3 normal{frame(0, 2), frame(1, 2), frame(2, 2)};
            const float wet = std::max(.06f,
                d.wetness * (1.f - std::min(1.f, d.age / d.lifetime)));
            result.push_back({
                math::inverse_affine(frame), math::direction(normal),
                wet, d.variant, d.pool
            });
        }
        return result;
    }

    void shutdown()
    {
        delete data;
        data = nullptr;
    }
}
