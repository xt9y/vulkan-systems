#pragma once
#include "../material/properties.hpp"
#include <vk/util/math/matrix.hpp>
#include <array>
#include <cstdint>
#include <string>

namespace fld
{
    using vec3=vk::util::math::vec3;
    using vec4=vk::util::math::vec4;
    using mat4=vk::util::math::mat4;

    // GPU std430 layout; positions/velocities remain resident on the GPU.
    struct alignas(16) particle
    {
        vec4 position{};   // xyz metres, w physical particle radius
        vec4 velocity{};   // xyz metres/second, w unused
        vec4 previous{};   // xyz prior substep position
        vec4 physical{};   // volume m^3, mass kg, remainder reserved
    };
    static_assert(sizeof(particle)==64);

    struct options
    {
        properties liquid{};
        std::uint32_t max_particles=2048;
        std::uint32_t pressure_iterations=4;
        std::uint32_t substeps=2;
        float spacing=.045f;
        float smoothing_radius=.102f;
        float floor_height=0.f;
        vec3 gravity{0.f,-9.81f,0.f};
    };

    struct drop
    {
        vec3 center{0.f,2.5f,-1.5f};
        float radius=.28f;
        vec3 velocity{};
    };

    // Each path is a SPIR-V compute shader compiled by the host's build.
    // clear, predict, build, lambda, delta, apply, finalize,
    // surface-normal estimate, capillary force.
    using shader_paths=std::array<std::string,9>;
}
