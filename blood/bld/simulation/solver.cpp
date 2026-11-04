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
        const vec3 bullet = math::direction(hit.direction, {0, 0, 1});
        // The blood exits the wound OUTWARDS. Starting inside the actor and
        // continuing in the bullet direction caused the old instant collisions
        // and made the actual spray almost invisible.
        vec3 outward = math::direction(hit.normal, bullet * -1.f);
        if (dot(outward, bullet) > 0.f) outward = outward * -1.f;
        const vec3 guide = std::abs(outward.y) < .94f ?
            vec3{0, 1, 0} : vec3{1, 0, 0};
        const vec3 side = math::direction(cross(outward, guide), {1, 0, 0});
        const vec3 up = cross(side, outward);
        const float energy = std::clamp(hit.energy, .3f, 2.2f);
        const std::uint32_t count = std::uint32_t(
            std::clamp(14.f + energy * 14.f, 16.f, 48.f));
        const vec3 center = hit.position + outward * .035f;
        for (std::uint32_t i = 0; i < count; ++i)
        {
            // Unequal connected streams separated by larger ballistic drops.
            const float phase = math::unit(seed) * 6.2831853f;
            const float spread = std::sqrt(math::unit(seed)) *
                (.12f + .29f * energy);
            const vec3 travel = math::direction(
                outward * (1.0f - .08f * math::unit(seed)) +
                side * (std::cos(phase) * spread) +
                up * (std::sin(phase) * spread), outward);
            const float speed = (.95f + math::unit(seed) * 4.5f) * energy;
            const float source_jitter = .008f * math::unit(seed);
            drop d;
            d.position = center +
                side * std::cos(phase) * source_jitter +
                up * std::sin(phase) * source_jitter;
            d.previous = d.position;
            d.velocity = travel * speed;
            d.radius = .009f + math::unit(seed) * .011f;
            // Larger deposits remain cohesive, smaller ones separate quickly.
            if ((i % 7u) == 0u) d.radius *= 1.4f;
            d.lifetime = 1.1f + math::unit(seed) * 1.5f;
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
        const float range = .105f;
        const float range2 = range * range;
        std::map<cell, std::vector<std::size_t>> grid;
        for (std::size_t i = 0; i < particles.size(); ++i)
            grid[quantize(particles[i].position, range)].push_back(i);

        struct neighbor
        {
            std::size_t a, b;
            vec3 direction;
            float distance, weight;
        };
        std::vector<neighbor> neighbors;
        std::vector<float> density(particles.size(), 1.f);

        for (std::size_t a = 0; a < particles.size(); ++a)
        {
            const auto c = quantize(particles[a].position, range);
            for (int z = -1; z <= 1; ++z)
                for (int y = -1; y <= 1; ++y)
                    for (int x = -1; x <= 1; ++x)
                    {
                        const auto found = grid.find({c.x+x, c.y+y, c.z+z});
                        if (found == grid.end()) continue;
                        for (const auto b : found->second)
                        {
                            if (b <= a || particles[a].group != particles[b].group)
                                continue;
                            const auto delta = particles[b].position -
                                               particles[a].position;
                            const float r2 = length_squared(delta);
                            if (r2 >= range2 || r2 <= 1e-10f) continue;
                            const float r = std::sqrt(r2);
                            const float q = 1.f - r2 / range2;
                            const float weight = q*q*q;
                            density[a] += weight;
                            density[b] += weight;
                            neighbors.push_back({a, b, delta / r, r, weight});
                        }
                    }
        }

        // SPH-style local density/pressure plus pairwise momentum-conserving
        // XSPH viscosity and surface attraction. This is bounded and stable
        // for the short-lived sprays; it is not a full incompressible solver.
        std::vector<vec3> forces(particles.size());
        for (const auto &n : neighbors)
        {
            const auto &a = particles[n.a];
            const auto &b = particles[n.b];
            const float pressure = (std::max(0.f, density[n.a] - 5.f) +
                                    std::max(0.f, density[n.b] - 5.f)) *
                                   n.weight * 2.4f;
            const float contact = a.radius + b.radius;
            const float repulsion = std::max(0.f, contact - n.distance) /
                                    std::max(.001f, contact) * 38.f;
            const float cohesion = data->options.cohesion * n.weight * 3.f;
            const auto force = (b.velocity - a.velocity) *
                               (data->options.viscosity * n.weight) +
                               n.direction * (cohesion - pressure - repulsion);
            forces[n.a] = forces[n.a] + force;
            forces[n.b] = forces[n.b] - force;
        }

        for (std::size_t i = 0; i < particles.size(); ++i)
        {
            auto &d = particles[i];
            d.previous = d.position;
            // Cap accelerations so dense impacts do not explode numerically.
            const auto force_length = length(forces[i]);
            const auto force = force_length > 110.f ?
                forces[i] * (110.f / force_length) : forces[i];
            d.velocity = d.velocity +
                (force + vec3{0, -data->options.gravity, 0} -
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
