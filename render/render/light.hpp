#pragma once
#include <vk/vk.hpp>
#include <cstdint>

namespace render
{
    // Optional packed light payload for user shaders; no enforced light model.
    enum class light_type:std::uint32_t {directional=0,point=1,spot=2};

    struct alignas(16) light
    {
        vk::util::math::vec4 position_range{};
        vk::util::math::vec4 direction_type{};
        vk::util::math::vec4 color_intensity{};
        vk::util::math::vec4 parameters{};
    };
    static_assert(sizeof(light)==64);
}
