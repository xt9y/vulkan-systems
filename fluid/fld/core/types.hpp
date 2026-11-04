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

    // Renderer-neutral optical parameters stored with each fluid phase.
    // All color coefficients are linear-light, inverse metre.
    struct appearance
    {
        vec3 absorption{.12f,.045f,.022f};
        vec3 scattering{.005f,.016f,.024f};
        float index_of_refraction=1.333f;
        float roughness=.035f;
        vec3 light_direction{.45f,.85f,.25f};
    };

    struct surface_properties
    {
        // Dimensionless fraction and physically meaningful contact angle.
        float porosity=0.f;                 // fraction 0..1
        float wetting=0.65f;                // adhesion 0..1
        float contact_angle=1.1f;          // radians
        float restitution=.04f;            // 0..1
        float friction=.50f;               // 0..1
        float roughness=.40f;              // normalized height-scale proxy
    };

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
        appearance optical{};
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

    // One generic liquid-impact source. Position/velocity are physical
    // initial conditions supplied by the host (metres, metres/second).
    // Material, surface tension, wetting and viscosity determine ejection.
    struct splash_source
    {
        vec3 position{}, normal{0,1,0}, velocity{};
        float volume=.000001f;          // m^3
        float lifetime=5.f;
        std::uint32_t phase=0, group=1, seed=1;
        surface_properties surface{};
    };

    struct splash_result
    {
        std::uint32_t particles=0;
        float retained_volume=0, emitted_volume=0;
        float weber=0, reynolds=0;
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
        // When true, the generic solver computes wetting, absorption and
        // rebound from the fluid and the receiving material. No blood code.
        bool material_response=true;
        surface_properties surface{};
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
        double emitted_mass_kg = 0;
        double deposited_mass_kg = 0;
        double expired_mass_kg = 0;
        double live_mass_kg = 0;
    };
}
