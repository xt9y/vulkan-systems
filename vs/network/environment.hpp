#pragma once
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <vs/scene/environment.hpp>

namespace vs::network::environment
{
// Versioned, endian-independent environment transport. No renderer,
// atmosphere model, game-specific packet type or platform dependency.
inline constexpr std::size_t packet_size = 68;

using packet = std::array<std::uint8_t, packet_size>;

namespace detail
{
inline void write(packet &dst, std::size_t at, std::uint64_t value, int width = 8)
{
    for (int i = 0; i < width; ++i)
        dst[at + i] = std::uint8_t(value >> (i * 8));
}

inline std::uint64_t read(std::span<const std::uint8_t> src,
                          std::size_t at, int width = 8)
{
    std::uint64_t value = 0;

    for (int i = 0; i < width; ++i)
        value |= std::uint64_t(src[at + i]) << (i * 8);

    return value;
}

inline void scalar(packet &dst, std::size_t at, float value)
{
    write(dst, at, std::bit_cast<std::uint32_t>(value), 4);
}

inline float scalar(std::span<const std::uint8_t> p, std::size_t at)
{
    return std::bit_cast<float>(std::uint32_t(read(p, at, 4)));
}

inline bool valid(const ::vs::scene::environment_state &e)
{
    if (!e.revision || !std::isfinite(e.time_seconds) ||
        !std::isfinite(e.exposure) || e.exposure <= 0.f ||
        !std::isfinite(e.sun.angular_radius) ||
        e.sun.angular_radius < 0.f || e.sun.angular_radius > 0.5f)
        return false;

    float length2 = 0.f;

    for (int i = 0; i < 3; ++i)
    {
        const auto d = e.sun.direction[i], r = e.sun.irradiance[i],
                   sky = e.ambient_irradiance[i];

        if (!std::isfinite(d) || !std::isfinite(r) || !std::isfinite(sky) ||
            r < 0.f || sky < 0.f) return false;
        length2 += d * d;
    }

    return !e.sun.enabled || (length2 > 0.9f && length2 < 1.1f);
}
} // namespace detail
inline packet encode(const ::vs::scene::environment_state &e)
{
    if (!detail::valid(e)) throw std::invalid_argument("Invalid environment state");
    packet p{};
    p[0] = 'V';
    p[1] = 'S';
    p[2] = 'E';
    p[3] = '1';
    detail::write(p, 4, e.revision);
    detail::write(p, 12, std::bit_cast<std::uint64_t>(e.time_seconds));

    for (int i = 0; i < 3; ++i)
    {
        detail::scalar(p, 20 + 4 * i, e.sun.direction[i]);
        detail::scalar(p, 32 + 4 * i, e.sun.irradiance[i]);
        detail::scalar(p, 52 + 4 * i, e.ambient_irradiance[i]);
    }

    detail::scalar(p, 44, e.sun.angular_radius);
    detail::write(p, 48, e.sun.enabled ? 1u : 0u, 4);
    detail::scalar(p, 64, e.exposure);

    return p;
}

inline std::optional<::vs::scene::environment_state>
decode(std::span<const std::uint8_t> bytes)
{
    if (bytes.size() != packet_size ||
        bytes[0] != 'V' || bytes[1] != 'S' || bytes[2] != 'E' || bytes[3] != '1')
        return std::nullopt;
    ::vs::scene::environment_state e;
    e.revision = detail::read(bytes, 4);
    e.time_seconds = std::bit_cast<double>(detail::read(bytes, 12));

    for (int i = 0; i < 3; ++i)
    {
        e.sun.direction[i] = detail::scalar(bytes, 20 + 4 * i);
        e.sun.irradiance[i] = detail::scalar(bytes, 32 + 4 * i);
        e.ambient_irradiance[i] = detail::scalar(bytes, 52 + 4 * i);
    }

    e.sun.angular_radius = detail::scalar(bytes, 44);

    const auto enabled = detail::read(bytes, 48, 4);

    if (enabled > 1u) return std::nullopt;
    e.sun.enabled = enabled != 0;
    e.exposure = detail::scalar(bytes, 64);

    return detail::valid(e) ? std::optional{e} : std::nullopt;
}
} // namespace vs::network::environment
