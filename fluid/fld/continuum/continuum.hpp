#pragma once
#include "../material/properties.hpp"
#include <vk/vk.hpp>
#include <vk/util/math/matrix.hpp>
#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace fld::continuum
{
    using vec3=vk::util::math::vec3;
    using vec4=vk::util::math::vec4;
    using mat4=vk::util::math::mat4;

    struct alignas(16) box
    {
        vec4 minimum{},maximum{};
    };
    static_assert(sizeof(box)==32);

    struct options
    {
        properties liquid{};
        std::uint32_t width=64,height=40,depth=64;
        float cell_size=.1f;
        vec3 origin{-3.2f,0.f,-3.2f};
        vec3 gravity{0.f,-9.81f,0.f};
        std::uint32_t pressure_iterations=36;
        std::uint32_t substeps=3;
        float render_scale=.5f;
        std::vector<box> colliders;
    };

    struct drop
    {
        vec3 center{.9f,2.75f,.75f};
        float radius=.68f;
    };

    // Compute variants: seed, advect velocity, curvature, divergence,
    // Jacobi pressure, projection, conservative volume-fraction advection.
    struct shader_paths
    {
        std::array<std::string,7> simulation{};
        std::array<std::string,2> rendering{}; // surface, upscale
    };

    struct render_input
    {
        vk::ImageView depth{},scene_color{},output{};
        vk::Sampler sampler{};
        mat4 view_projection=mat4::identity();
        std::uint32_t width=0,height=0;
        vk::ImageLayout depth_layout=vk::ImageLayout::eDepthStencilReadOnlyOptimal;
        vk::ImageLayout color_layout=vk::ImageLayout::eGeneral;
        bool reversed_depth=false;
        vec3 debug_color{.04f,1.f,.06f};
        float debug_opacity=.85f;
    };

    // Generic GPU continuum liquid: a staggered MAC velocity field,
    // pressure projection and conservative VOF advection. No particles.
    void init(const options& settings,const drop& initial,
              const shader_paths& shaders,std::uint32_t frames,
              std::uint32_t screen_width,std::uint32_t screen_height);
    bool active();
    void record(vk::CommandBuffer command,std::uint32_t image,
                float elapsed_seconds,const render_input& target);
    void shutdown();
}
