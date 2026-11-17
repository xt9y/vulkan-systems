#pragma once
#include <vs/scene/scene.hpp>
#include <array>
#include <cstdint>
#include <string_view>
#include <span>
#include <vector>
#include <vk/vk.hpp>
namespace vs::animation {
using matrix=std::array<float,16>;
void synchronize(std::span<const scene::skin> skins);
std::uint32_t joints(std::uint32_t skin_id);
bool sample(std::uint32_t skin_id,std::uint32_t clip,float seconds,
            std::vector<matrix>& palette,std::uint32_t overlay=UINT32_MAX,
            float weight=0.f,std::uint32_t mask_root=UINT32_MAX,bool loop=true);
float duration(std::uint32_t skin_id,std::uint32_t clip);
std::uint32_t joint(std::uint32_t skin_id,std::string_view name);
bool joint_transform(std::uint32_t skin_id,std::uint32_t index,matrix& out);
bool offset_joint(std::uint32_t skin_id,std::uint32_t root,
                  vk::util::math::vec3 offset,std::vector<matrix>& palette);
void reset();
}
