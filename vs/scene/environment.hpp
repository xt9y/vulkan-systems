#pragma once
#include <array>
#include <cstdint>

namespace vs::scene {
// Shared input contract only: no sky model, LUTs, weather or sun renderer.
// Physical irradiance is linear and independent of display exposure.
struct directional_light {
    std::array<float,3> direction{0.f,-1.f,0.f};
    std::array<float,3> irradiance{0.f,0.f,0.f};
    float angular_radius=0.00465f;
    bool enabled=false;
};
struct environment_state {
    std::uint64_t revision=0;
    double time_seconds=0.0;
    directional_light sun{};
    std::array<float,3> ambient_irradiance{};
    float exposure=1.f;
};
} // namespace vs::scene
