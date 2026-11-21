#pragma once
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>

namespace vs::spatial
{
// Region addresses are exact even when world coordinates exceed float precision.
// Every region contains a fixed, 256-metre local coordinate space.
inline constexpr double region_extent = 256.0;
struct region_id
{
    std::int64_t x = 0, y = 0, z = 0;

    bool operator==(const region_id &) const = default;
};

struct region_hash
{
    std::size_t operator()(region_id id) const noexcept
    {
        auto mix = [](std::uint64_t v)
        {
            v ^= v >> 30;
            v *= 0xbf58476d1ce4e5b9ull;
            v ^= v >> 27;
            v *= 0x94d049bb133111ebull;

            return v ^ (v >> 31);
        };

        const auto a = mix(static_cast<std::uint64_t>(id.x));
        const auto b = mix(static_cast<std::uint64_t>(id.y));
        const auto c = mix(static_cast<std::uint64_t>(id.z));

        return static_cast<std::size_t>(mix(a ^ b ^ (c << 1)));
    }
};

struct world_position
{
    region_id region{};
    std::array<double, 3> local{};
};

inline std::int64_t checked_region(long double value)
{
    if (!std::isfinite(value) ||
        value < static_cast<long double>(std::numeric_limits<std::int64_t>::min()) ||
        value >= -static_cast<long double>(std::numeric_limits<std::int64_t>::min()))
        throw std::out_of_range("World region coordinate overflow");

    return static_cast<std::int64_t>(value);
}
// Preserve adjacent region indices even on platforms where long double
// has the same 53-bit mantissa as double (including Apple Silicon).
inline std::int64_t shift_region(std::int64_t region, std::int64_t delta)
{
    if ((delta > 0 && region > std::numeric_limits<std::int64_t>::max() - delta) ||
        (delta < 0 && region < std::numeric_limits<std::int64_t>::min() - delta))
        throw std::out_of_range("World region coordinate overflow");

    return region + delta;
}

inline std::int64_t integer_shift(std::int64_t region, long double delta)
{
    constexpr auto lo = static_cast<long double>(
        std::numeric_limits<std::int64_t>::min());
    constexpr auto hi = -lo;

    if (!std::isfinite(delta) || delta < lo || delta >= hi)
        throw std::out_of_range("World region displacement overflow");

    return shift_region(region, static_cast<std::int64_t>(delta));
}

inline world_position normalize(world_position value)
{
    auto coordinate = [&](std::int64_t &region, double &local)
    {
        if (!std::isfinite(local)) throw std::invalid_argument("Invalid world position");

        const auto delta = std::floor(static_cast<long double>(local) / region_extent);
        region = integer_shift(region, delta);
        local -= static_cast<double>(delta * region_extent);
        // Compensate for a rounded local result at the tile boundary.
        if (local < 0.0) {
            region = shift_region(region, -1);
            local += region_extent;
        }
        else if (local >= region_extent) {
            region = shift_region(region, 1);
            local -= region_extent;
        }
    };

    coordinate(value.region.x, value.local[0]);
    coordinate(value.region.y, value.local[1]);
    coordinate(value.region.z, value.local[2]);

    return value;
}

inline world_position from_world(std::array<double, 3> absolute)
{
    world_position result{};
    result.local = absolute;

    return normalize(result);
}

inline world_position translated(world_position position, std::array<double, 3> delta)
{
    for (int i = 0; i < 3; ++i)
        position.local[i] += delta[i];

    return normalize(position);
}

inline std::array<float, 3> camera_relative(const world_position &position,
                                            const world_position &origin)
{
    const auto exact_delta = [](std::int64_t a, std::int64_t b)
    {
        if (a >= b) return static_cast<long double>(
            std::uint64_t(a) - std::uint64_t(b));
        return -static_cast<long double>(
            std::uint64_t(b) - std::uint64_t(a));
    };

    const std::array<long double, 3> region_delta{
        exact_delta(position.region.x, origin.region.x),
        exact_delta(position.region.y, origin.region.y),
        exact_delta(position.region.z, origin.region.z)
    };

    std::array<float, 3> result{};

    for (int i = 0; i < 3; ++i)
    {
        const auto value = region_delta[i] * region_extent +
                           static_cast<long double>(position.local[i]) - origin.local[i];

        if (!std::isfinite(value) ||
            std::abs(value) > std::numeric_limits<float>::max())
            throw std::out_of_range("Camera relative offset overflow");
        result[i] = static_cast<float>(value);
    }

    return result;
}
} // namespace vs::spatial
