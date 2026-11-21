#pragma once
#include <cstdint>
#include <filesystem>
#include <vector>
#include <vs/scene/scene.hpp>
namespace vs::model
{
struct ids
{ std::uint32_t mesh = 1, instance = 1, texture = 1, skin = 1; };
// A model import is an asset package, not a game world.
// The application decides where to store its instances and components.
using data = ::vs::scene::data; // Model import and renderer scene share one type.
data load(const std::filesystem::path &path, ids first = {});
} // namespace vs::model
