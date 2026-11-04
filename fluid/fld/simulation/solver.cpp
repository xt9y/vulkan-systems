#include "solver.hpp"
#include "grid.hpp"
#include "../math/kernels.hpp"
#include "../material/rheology.hpp"
#include "../material/contact.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace fld::simulation
{
    namespace
    {
        struct workspace
        {
            std::vector<float> density, lambda, square;
            std::vector<vec3> sum_gradient, correction, viscosity;
            std::vector<pair> pairs;

            explicit workspace(std::size_t count) :
                density(count), lambda(count), square(count),
                sum_gradient(count), correction(count), viscosity(count) {}
        };

        float rest_density(const system::state &world, const particle &p)
        {
            return world.materials.at(p.phase).density;
        }

        void densities(const system::state &world, workspace &s, float h)
        {
            const auto &particles = world.particles;
            std::fill(s.square.begin(), s.square.end(), 0.f);
            std::fill(s.sum_gradient.begin(), s.sum_gradient.end(), vec3{});
            for (std::size_t i = 0; i < particles.size(); ++i)
            {
                // The self-kernel is required: surface particles have
                // fewer neighbors but must still carry mass.
                s.density[i] = particles[i].mass * math::poly6(0.f, h);
            }

            for (const auto [a, b] : s.pairs)
            {
                const auto &pa = particles[a], &pb = particles[b];
                const vec3 delta = pa.predicted - pb.predicted;
                const float r2 = vk::util::math::length_squared(delta);
                const float weight = math::poly6(r2, h);
                const vec3 gradient = math::spiky_gradient(delta, h);
                s.density[a] += pb.mass * weight;
                s.density[b] += pa.mass * weight;

                // Derivatives of Ci=rho_i/rho0-1 for both pair members.
                const vec3 ga = gradient * (pb.mass / rest_density(world, pa));
                const vec3 gb = gradient * (pa.mass / rest_density(world, pb));
                s.sum_gradient[a] = s.sum_gradient[a] + ga;
                s.sum_gradient[b] = s.sum_gradient[b] - gb;
                s.square[a] += vk::util::math::length_squared(ga);
                s.square[b] += vk::util::math::length_squared(gb);
            }
        }

        void constraints(system::state &world, workspace &s, float h)
        {
            const auto &settings = world.settings;
            const auto &particles = world.particles;
            std::fill(s.correction.begin(), s.correction.end(), vec3{});
            float largest = 0.f;

            for (std::size_t i = 0; i < particles.size(); ++i)
            {
                const float deviation = s.density[i] /
                    rest_density(world, particles[i]) - 1.f;
                largest = std::max(largest, std::abs(deviation));

                // Compressibility correction; negative constraints at the
                // free surface should not create artificial expansion.
                const float C = std::max(0.f, deviation);
                const float denominator =
                    s.square[i] +
                    vk::util::math::length_squared(s.sum_gradient[i]) +
                    settings.pressure_epsilon;
                s.lambda[i] = -C / denominator;
            }
            world.stats.maximum_density_error = largest;

            const float denominator = std::max(
                .000001f, math::poly6(.3f*.3f*h*h, h));
            for (const auto [a, b] : s.pairs)
            {
                const auto &pa = particles[a], &pb = particles[b];
                const vec3 r = pa.predicted - pb.predicted;
                const float r2 = vk::util::math::length_squared(r);
                const float ratio = math::poly6(r2, h) / denominator;
                const float q = ratio * ratio;
                const float scorr =
                    -settings.tensile_correction * q * q;
                const vec3 gradient = math::spiky_gradient(r, h);
                const float factor = s.lambda[a] + s.lambda[b] + scorr;
                s.correction[a] = s.correction[a] +
                    gradient * (factor * pb.mass / rest_density(world, pa));
                s.correction[b] = s.correction[b] -
                    gradient * (factor * pa.mass / rest_density(world, pb));
            }

            // Independent Jacobi correction avoids iteration-order bias.
            // Limit large corrections to control edge instabilities.
            const float limit = settings.spacing * .32f;
            for (std::size_t i = 0; i < particles.size(); ++i)
            {
                vec3 correction = s.correction[i];
                const auto length = math::norm(correction);
                if (length > limit) correction = correction * (limit / length);
                world.particles[i].predicted =
                    world.particles[i].predicted + correction;
            }
        }

        void viscous_response(system::state &world, workspace &s, float h,
                              float dt)
        {
            auto &particles = world.particles;
            std::fill(s.viscosity.begin(), s.viscosity.end(), vec3{});

            for (const auto [a, b] : s.pairs)
            {
                const auto &pa = particles[a], &pb = particles[b];
                const auto separation = pa.predicted - pb.predicted;
                const float distance = math::norm(separation);
                if (distance >= h || distance < 1e-7f) continue;

                const float shear = math::norm(pa.velocity - pb.velocity) /
                                    std::max(distance, .0001f);
                const auto &material = world.materials[pa.phase];
                const float eta = rheology::viscosity(material, shear);
                const float nu = eta / material.density;
                const float laplacian = math::viscosity_laplacian(distance, h);
                const auto relative_velocity = pb.velocity - pa.velocity;

                const auto delta_a = relative_velocity * (
                    nu * dt * laplacian * pb.mass / material.density);
                const auto delta_b = relative_velocity * (
                    nu * dt * laplacian * pa.mass / material.density);
                s.viscosity[a] = s.viscosity[a] + delta_a;
                s.viscosity[b] = s.viscosity[b] - delta_b;

                // Symmetric cohesive acceleration based on surface tension.
                // Low-energy pair cohesion is independent of screen-space
                // appearance and conserves pairwise linear momentum.
                const float q = std::max(0.f, 1.f-distance/h);
                const float acceleration = material.surface_tension /
                    (material.density * h*h) * q*q * dt;
                const auto direction = separation / distance;
                s.viscosity[a] = s.viscosity[a] -
                    direction * (acceleration * pb.mass / pa.mass);
                s.viscosity[b] = s.viscosity[b] +
                    direction * (acceleration * pa.mass / pb.mass);
            }

            for (std::size_t i = 0; i < particles.size(); ++i)
            {
                auto correction = s.viscosity[i];
                const float limit = 2.f;
                const float length = math::norm(correction);
                if (length > limit)
                    correction = correction * (limit / length);
                particles[i].velocity = particles[i].velocity + correction;
            }
        }
    }

    void solve(system::state &world, float dt,
               const collision_callback &collision,
               const contact_callback &contact)
    {
        auto &ps = world.particles;
        if (ps.empty()) return;
        const std::size_t count = ps.size();
        const float h = world.settings.smoothing_radius;
        workspace s(count);

        for (auto &p : ps)
        {
            p.previous = p.position;
            p.velocity = p.velocity +
                (vec3{0.f, -world.settings.gravity, 0.f} -
                 p.velocity * world.settings.air_drag) * dt;
            p.predicted = p.position + p.velocity * dt;
        }

        for (std::uint32_t k = 0; k < world.settings.pressure_iterations; ++k)
        {
            s.pairs = neighbors(ps, h);
            densities(world, s, h);
            constraints(world, s, h);
        }
        world.stats.neighbor_pairs = std::uint32_t(s.pairs.size());

        for (auto &p : ps)
        {
            p.velocity = (p.predicted - p.previous) / dt;
        }
        // Rebuild because the final density correction moved particles.
        s.pairs = neighbors(ps, h);
        viscous_response(world, s, h, dt);

        std::vector<std::uint8_t> consumed(count, 0);
        for (std::size_t i = 0; i < count; ++i)
        {
            auto &p = ps[i];
            if (collision)
            {
                auto hit = collision(p.previous, p.predicted, p.radius);
                if (hit.hit)
                {
                    if (hit.material_response)
                        wetting::resolve(world.materials.at(p.phase),p,hit);
                    ++world.stats.contacts;
                    const float deposited = hit.absorb ? 1.f :
                        std::clamp(hit.deposition_fraction, 0.f, 1.f);
                    if (contact && deposited > 0.f)
                    {
                        auto deposited_mass = p;
                        deposited_mass.mass *= deposited;
                        contact(deposited_mass, hit);
                    }
                    const float original_mass=p.mass;
                    p.mass *= (1.f-deposited);
                    if(original_mass>0.f)
                        p.radius*=std::cbrt(std::max(0.f,p.mass)/original_mass);
                    ++p.contacts;
                    const float minimum_mass =
                        rest_density(world,p) *
                        std::pow(world.settings.spacing,3.f) * .03f;
                    if (hit.absorb || p.mass <= minimum_mass)
                    {
                        consumed[i] = 1;
                    }
                    else
                    {
                        const auto normal = math::safe_normal(hit.normal);
                        p.predicted = hit.position +
                            normal * (p.radius + .0002f);
                        const float vn = vk::util::math::dot(p.velocity, normal);
                        if (vn < 0.f)
                            p.velocity = p.velocity -
                                normal * (vn * (1.f + std::clamp(
                                    hit.restitution, 0.f, 1.f)));
                        const float remaining = std::clamp(hit.friction, 0.f, 1.f);
                        const float normal_speed =
                            vk::util::math::dot(p.velocity, normal);
                        p.velocity = normal * normal_speed +
                            (p.velocity - normal*normal_speed) *
                            (1.f - remaining);
                    }
                }
            }
            p.position = p.predicted;
            p.age += dt;
        }

        std::size_t write = 0;
        for (std::size_t i = 0; i < count; ++i)
        {
            if (consumed[i] || ps[i].age >= ps[i].lifetime) continue;
            if (!std::isfinite(ps[i].position.x) ||
                !std::isfinite(ps[i].position.y) ||
                !std::isfinite(ps[i].position.z))
                continue;
            if (write != i) ps[write] = ps[i];
            ++write;
        }
        ps.resize(write);
    }
}
