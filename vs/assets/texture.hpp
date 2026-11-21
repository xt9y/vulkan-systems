#pragma once
#include <cstdint>
#include <span>
#include <vector>
#include <vs/scene/scene.hpp>

namespace vs::assets
{
// One texture decoder for raster asset uploads and optics PBR atlases.
// Borrow already-decoded RGBA without copying; otherwise decode into scratch.
// The returned span is valid while image and scratch remain alive.
std::span<const std::uint8_t> rgba(const scene::texture &image,
                                   std::vector<std::uint8_t> &scratch);
} // namespace vs::assets
