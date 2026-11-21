#pragma once
#include <memory>
#include <span>
#include <vector>
#include <vs/scene/scene.hpp>
#include <vs/spatial/regions.hpp>

namespace vs::scene
{
// A spatially scoped scene fragment. Mesh/texture/skin asset definitions
// remain separate and may be referenced by many regions or instances.
struct region_fragment
{
    std::vector<instance> instances;
    std::vector<surface> surfaces;
    std::vector<collision_box> collision_boxes;
    std::vector<point_light> lights;
};

using region_catalog = ::vs::spatial::region_store<region_fragment>;
} // namespace vs::scene
