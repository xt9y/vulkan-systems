#pragma once
#include <array>
#include <vk/vk.hpp>

namespace render
{
    // Optional payload only; shaders may use any camera format or none.
    struct camera
    {
        std::array<float,16> view{};
        std::array<float,16> projection{};
        std::array<float,16> view_projection{};
        vk::util::math::vec4 world_position{};
    };
}
