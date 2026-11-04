#include "solver.hpp"
#include "../math/geometry.hpp"
#include "../surfaces/decals.hpp"
#include "../surfaces/wounds.hpp"
#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>

namespace bld::simulation
{
    namespace
    {
        state *data = nullptr;
        struct cell
        {
            int x, y, z;
            bool operator<(const cell &other) const
            {
                if (x != other.x) return x < other.x;
                if (y != other.y) return y < other.y;
                return z < other.z;
            }
        };

        cell quantize(vec3 p, float size)
        {
            return {int(std::floor(p.x / size)),
                    int(std::floor(p.y / size)),
                    int(std::floor(p.z / size))};
        }

        void insert(const drop &d)
        {
            if (!data || !data->options.max_particles) return;
            if (data->particles.size() >= data->options.max_particles)
                data->particles.erase(data->particles.begin());
            data->particles.push_back(d);
        }
    }

    state *get() { return data; }

    void init(const settings &options)
    {
        if (data) throw std::logic_error("Blood simulation already initialized");
        data = new state{};
        data->options = options;
        data->particles.reserve(options.max_particles);
    }

    void emit(const impact &hit)
    {
        if (!data || hit.type != material::flesh) return;
        using namespace vk::util::math;
        auto seed = hit.seed ? hit.seed : 1;
        const vec3 axis = math::direction(hit.direction, {0, 0, 1});
        const vec3 guide = std::abs(axis.y) < .94f ?
            vec3{0, 1, 0} : vec3{1, 0, 0};
        const vec3 side = math::direction(cross(axis, guide), {1, 0, 0});
        const vec3 up = cross(side, axis);
        const float energy = std::max(.2f, std::min(hit.energy, 2.5f));
        const std::uint32_t count = std::uint32_t(
            std::max(6.f, std::min(18.f, 7.f + energy * 4.f)));
        for (std::uint32_t i = 0; i < count; ++i)
        {
            const float a = math::unit(seed) * 6.2831853f;
            const float spread = std::sqrt(math::unit(seed)) * (.12f + energy * .09f);
            const vec3 travel = math::direction(
                axis + side * (std::cos(a) * spread) +
                       up * (std::sin(a) * spread) +
                       math::direction(hit.normal) * .06f, axis);
            const float speed = (2.7f + math::unit(seed) * 4.3f) * energy;
            drop d;
            d.position = hit.position + axis * .035f;
            d.previous = d.position;
            d.velocity = travel * speed;
            d.radius = .011f + math::unit(seed) * .018f;
            d.lifetime = 1.25f + math::unit(seed) * 1.4f;
            d.seed = math::random(seed);
            d.group = hit.id;
            insert(d);
        }
    }

    void drip(vec3 position, std::uint32_t seed)
    {
        if (!data) return;
        drop d;
        d.position = position;
        d.previous = position;
        d.radius = .012f;
        d.velocity = {(.5f-math::unit(seed))*.25f, -.1f,
                      (.5f-math::unit(seed))*.25f};
        d.lifetime = 2.3f;
        d.seed = math::random(seed);
        insert(d);
    }

    void update(float dt, const collision_callback &world)
    {
        if (!data || data->particles.empty()) return;
        using namespace vk::util::math;
        auto &particles = data->particles;
        const float range = .13f;
        const float range2 = range * range;
        std::map<cell, std::vector<std::size_t>> grid;
        for (std::size_t i = 0; i < particles.size(); ++i)
            grid[quantize(particles[i].position, range)].push_back(i);

        std::vector<vec3> forces(particles.size());
        // Short-range viscosity and cohesion keep a burst looking like liquid
        // instead of unrelated ballistic pellets. Cells bound the neighbor work.
        for (std::size_t i = 0; i < particles.size(); ++i)
        {
            const auto c = quantize(particles[i].position, range);
            for (int z = -1; z <= 1; ++z)
                for (int y = -1; y <= 1; ++y)
                    for (int x = -1; x <= 1; ++x)
                    {
                        auto found = grid.find({c.x+x, c.y+y, c.z+z});
                        if (found == grid.end()) continue;
                        for (auto j : found->second)
                        {
                            if (i == j || particles[i].group != particles[j].group)
                                continue;
                            const vec3 between = particles[j].position -
                                                 particles[i].position;
                            const float r2 = length_squared(between);
                            if (r2 < 1e-8f || r2 >= range2) continue;
                            const float r = std::sqrt(r2);
                            const float weight = 1.f - r / range;
                            // Viscous velocity sharing and surface attraction.
                            const float contact =
                                particles[i].radius + particles[j].radius;
                            const float overlap = std::max(0.f, contact - r);
                            const float repulsion = overlap / std::max(.001f, contact) *
                                (data->options.collision_smoothing * 30.f);
                            forces[i] = forces[i] +
                                (particles[j].velocity - particles[i].velocity) *
                                    (weight * data->options.viscosity) +
                                (between / r) *
                                    (weight * data->options.cohesion - repulsion);
                        }
                    }
        }

        for (std::size_t i = 0; i < particles.size(); ++i)
        {
            auto &d = particles[i];
            d.previous = d.position;
            d.velocity = d.velocity +
                (forces[i] + vec3{0, -data->options.gravity, 0} -
                 d.velocity * data->options.air_drag) * dt;
            const vec3 next = d.position + d.velocity * dt;
            if (world)
            {
                const auto hit = world(d.position, next);
                if (hit.hit)
                {
                    auto seed = d.seed;
                    if (hit.skinned && hit.uv_valid)
                    {
                        wounds::stamp(hit.target, hit.uv, .02f, .85f, seed);
                    }
                    else
                    {
                        const float size = .065f + math::unit(seed) * .085f;
                        const float stretch = 1.1f + math::unit(seed) * 1.5f;
                        surfaces::add(hit, size, seed, false, stretch);
                        if (hit.normal.y > .7f && math::unit(seed) > .5f)
                            surfaces::add(hit, .16f + math::unit(seed)*.14f,
                                          seed, true);
                    }
                    d.age = d.lifetime;
                    continue;
                }
            }
            d.position = next;
            d.age += dt;
        }

        particles.erase(std::remove_if(particles.begin(), particles.end(),
            [](const drop &d) {
                return d.age >= d.lifetime || d.position.y < -10000.f;
            }), particles.end());
    }

    void shutdown()
    {
        delete data;
        data = nullptr;
    }
}
