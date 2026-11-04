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
        if (!data || !id) return;
        // Decals can arrive over the network before their owner transform.
        // Resolve their local pose exactly once, when that transform arrives.
        for (auto &decal : data->decals)
        {
            if (decal.target != id || decal.anchored) continue;
            decal.local = math::inverse_affine(model) * decal.world;
            decal.anchored = true;
        }
        data->transforms[id] = model;
    }

    void set_skinned(instance_id id, bool skinned)
    {
        if (!data || !id) return;
        if (skinned) data->animated.insert(id);
        else data->animated.erase(id);
    }

    bool is_skinned(instance_id id)
    {
        return data && data->animated.count(id) != 0;
    }

    void remove_transform(instance_id id)
    {
        if (!data) return;
        data->transforms.erase(id);
        data->animated.erase(id);
        auto &marks = data->decals;
        marks.erase(std::remove_if(marks.begin(), marks.end(),
            [id](const decal &mark) { return mark.target == id; }), marks.end());
    }

    void add(const collision &hit, float size, std::uint32_t seed,
             bool pool, float stretch)
    {
        if (!hit.hit || hit.skinned || is_skinned(hit.target)) return;
        add(hit.target, hit.position, hit.normal, size, seed, pool, stretch);
    }

    void add(instance_id target, vec3 position, vec3 normal, float size,
             std::uint32_t seed, bool pool, float stretch)
    {
        if (!data || data->capacity == 0 || is_skinned(target)) return;
        if (pool)
        {
            // Nearby impacts feed an existing puddle rather than stacking
            // dozens of giant, identical overlapping red circles.
            for (auto &existing : data->decals)
            {
                if (!existing.pool || existing.target != target) continue;
                auto frame = existing.world;
                const auto owner = data->transforms.find(target);
                if (existing.anchored && owner != data->transforms.end())
                    frame = owner->second * existing.local;
                const auto center = math::position(frame);
                const auto delta = position - center;
                if (vk::util::math::length_squared(delta) > .12f * .12f)
                    continue;
                existing.final_radius = std::min(.55f,
                    existing.final_radius + std::max(.008f, size * .06f));
                existing.wetness = std::min(1.f, existing.wetness + .08f);
                existing.age = std::max(0.f, existing.age - .4f);
                return;
            }
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
        {
            d.local = math::inverse_affine(owner->second) * frame;
            d.anchored = true;
        }
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
            if (d.anchored && found != data->transforms.end())
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
                wet, d.variant, d.pool, d.target
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
