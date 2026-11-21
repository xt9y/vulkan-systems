#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <stdexcept>

namespace vs::optics
{
inline void copy_rgba8_rows_with_gutter(
    std::span<std::uint8_t> rows, std::uint32_t atlas_width,
    std::uint32_t first_row, std::uint32_t row_count,
    std::uint32_t x, std::uint32_t y, std::uint32_t width,
    std::uint32_t height, std::span<const std::uint8_t> texture)
{
    if (!atlas_width || !row_count || !width || !height ||
        std::uint64_t(x) + width + 2u > atlas_width ||
        std::size_t(width) > std::numeric_limits<std::size_t>::max() /
                                 std::size_t(height) / 4u ||
        std::size_t(atlas_width) > std::numeric_limits<std::size_t>::max() /
                                       std::size_t(row_count) / 4u ||
        texture.size() != std::size_t(width) * height * 4u ||
        rows.size() < std::size_t(atlas_width) * row_count * 4u)
        throw std::invalid_argument("optics: invalid atlas row copy");

    const std::size_t atlas_stride = std::size_t(atlas_width) * 4u;
    const std::size_t texture_stride = std::size_t(width) * 4u;

    const auto texture_end = std::uint64_t(y) + height + 1u;

    for (std::uint32_t row = 0; row < row_count; ++row)
    {
        const auto atlas_row = std::uint64_t(first_row) + row;

        if (atlas_row < y || atlas_row > texture_end) continue;

        const auto source_row = std::clamp<std::int64_t>(
            std::int64_t(atlas_row) - std::int64_t(y) - 1,
            0, std::int64_t(height) - 1);
        const auto *source = texture.data() +
                             std::size_t(source_row) * texture_stride;

        auto *destination = rows.data() + std::size_t(row) * atlas_stride +
                            std::size_t(x) * 4u;
        std::memcpy(destination, source, 4u);
        std::memcpy(destination + 4u, source, texture_stride);
        std::memcpy(destination + 4u + texture_stride,
                    source + texture_stride - 4u, 4u);
    }
}

inline void copy_rgba8_with_gutter(
    std::span<std::uint8_t> atlas, std::uint32_t atlas_width,
    std::uint32_t x, std::uint32_t y, std::uint32_t width,
    std::uint32_t height,
    std::span<const std::uint8_t> texture)
{
    const std::size_t atlas_stride = std::size_t(atlas_width) * 4u;
    const std::size_t texture_stride = std::size_t(width) * 4u;
    const std::size_t row_bytes = texture_stride + 8u;
    const std::size_t first_pixel =
        (std::size_t(y + 1u) * atlas_width + x + 1u) * 4u;

    for (std::uint32_t row = 0; row < height; ++row)
    {
        const auto *source = texture.data() + std::size_t(row) * texture_stride;
        auto *destination = atlas.data() + first_pixel + std::size_t(row) * atlas_stride;
        std::memcpy(destination - 4u, source, 4u);
        std::memcpy(destination, source, texture_stride);
        std::memcpy(destination + texture_stride,
                    source + texture_stride - 4u, 4u);
    }

    auto *first = atlas.data() + first_pixel - 4u;
    auto *last = first + std::size_t(height - 1u) * atlas_stride;
    std::memcpy(first - atlas_stride, first, row_bytes);
    std::memcpy(last + atlas_stride, last, row_bytes);
}
} // namespace vs::optics
