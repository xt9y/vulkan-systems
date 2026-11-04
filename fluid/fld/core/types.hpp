#pragma once
#include <vk/util/math/matrix.hpp>
#include <cstdint>
#include <functional>
#include <vector>

namespace fld
{
    using vec2 = vk::util::math::vec2;
    using vec3 = vk::util::math::vec3;
    using vec4 = vk::util::math::vec4;
    using mat4 = vk::util::math::mat4;

    struct material
    {
        // SI units. Defaults: water at about 20 Celsius.
        float density = 998.0f;             // kg/m^3
        float viscosity_zero = .001f;       // Pa*s
        float viscosity_infinite = .001f;   // Pa*s
        float relaxation = 0.0f;            // seconds
        float flow_index = 1.0f;            // Carreau-Yasuda exponent n
        float yasuda_exponent = 2.0f;       // Carreau-Yasuda exponent a
        float surface_tension = .072f;      // N/m
        float contact_angle = 1.2f;         // radians, receiver dependent
    };

    struct particle
    {
        vec3 position{}, velocity{};
        vec3 previous{}, predicted{};
        std::uint32_t phase = 0, group = 0;
        std::uint32_t contacts = 0;
        float mass = 0, radius = .0025f;
        float age = 0, lifetime = 8;
    };

    struct emission
    {
        vec3 position{}, velocity{};
        float volume = .00001f;              // m^3, 10 mL
        float spread = .25f;                 // m/s
        float lifetime = 8;
        std::uint32_t phase = 0, group = 1, seed = 1;
    };

    struct collision
    {
        bool hit = false, absorb = false;
        vec3 position{}, normal{0, 1, 0};
        std::uint32_t target = 0;
        vec2 uv{};
        bool uv_valid = false;
        float restitution = .02f, friction = .35f;
        // Fraction of this particle's mass transferred to the receiving
        // surface. The remaining mass continues after collision.
        float deposition_fraction = 0.f;
    };

    using collision_callback =
        std::function<collision(vec3 from, vec3 to, float particle_radius)>;
    using contact_callback =
        std::function<void(const particle &drop, const collision &impact)>;

    struct options
    {
        std::uint32_t max_particles = 8192;
        std::uint32_t pressure_iterations = 5;
        std::uint32_t max_substeps = 10;
        float spacing = .004f;             // particle separation in metres
        float smoothing_radius = .010f;
        float fixed_step = 1.f / 180.f;
        float gravity = 9.81f;
        float air_drag = .04f;
        float pressure_epsilon = 1e-5f;
        float tensile_correction = .001f;
        float vorticity = .0f;
    };

    struct statistics
    {
        std::uint64_t steps = 0, contacts = 0;
        std::uint32_t neighbor_pairs = 0;
        float maximum_density_error = 0;
    };
}
