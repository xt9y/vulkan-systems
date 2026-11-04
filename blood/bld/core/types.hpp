#pragma once
#include <vk/util/math/matrix.hpp>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace bld
{
    using vec2 = vk::util::math::vec2;
    using vec3 = vk::util::math::vec3;
    using vec4 = vk::util::math::vec4;
    using mat4 = vk::util::math::mat4;

    using instance_id = std::uint32_t;

    enum class material : std::uint8_t {
        default_surface, flesh, metal, stone, wood, glass
    };

    struct impact
    {
        std::uint64_t id = 0;
        std::uint32_t source = 0;
        instance_id target = 0;
        std::uint32_t seed = 1, tick = 0;
        material type = material::flesh;
        vec3 position{}, normal{0, 1, 0}, direction{0, 0, 1};
        vec2 uv{};
        bool uv_valid = false;
        float energy = 1;
    };

    struct collision
    {
        bool hit = false;
        instance_id target = 0;
        vec3 position{}, normal{0, 1, 0};
        vec2 uv{};
        bool uv_valid = false;
        bool skinned = false;
    };

    // Segment end points are in world space. The caller owns mesh/BVH/physics.
    using collision_callback = std::function<collision(vec3, vec3)>;

    struct settings
    {
        std::uint32_t max_particles = 512;
        std::uint32_t max_decals = 1024;
        std::uint32_t max_pools = 64;
        std::uint32_t max_wounds = 256;
        std::uint32_t max_instances = 512;
        std::uint32_t wound_resolution = 128;
        float gravity = 9.81f;
        float air_drag = 0.25f;
        float viscosity = 2.2f;
        float cohesion = 1.1f;
        float collision_smoothing = 0.12f;
        float fixed_step = 1.0f / 120.0f;
    };
}
