#pragma once

#include "../core/types.hpp"
#include "../porous/transport.hpp"
#include <cstdint>
#include <vector>

namespace fld::surface
{
    // Conservative finite-volume sheet on a UV chart. A cell's mobile and
    // pore volumes are m^3, not arbitrary mask opacity; all transport uses
    // equal and opposite face fluxes. The host can supply local geometry for
    // curved meshes, including normals and local tangent directions.
    struct cell
    {
        float mobile=0.f;
        float stored=0.f;
        float residue=0.f; // immobilized pigment/film, still conserves volume
    };

    struct chart_vertex
    {
        vec3 position{},normal{0.f,1.f,0.f};
        vec2 uv{};
    };

    struct sample
    {
        vec3 position{},normal{0.f,1.f,0.f};
        vec3 tangent_u{1.f,0.f,0.f},tangent_v{0.f,0.f,1.f};
        bool valid=false;
    };

    struct grid
    {
        std::uint32_t width=0,height=0;
        vec2 metres_per_uv{1.f,1.f};
        vec3 normal{0.f,1.f,0.f},tangent_u{1.f,0.f,0.f},
             tangent_v{0.f,0.f,1.f};
        mat4 transform=mat4::identity();
        bool open_edges=false;
        std::vector<cell> cells;
        std::vector<sample> geometry;  // optional triangle-mesh chart
        std::uint64_t generation=1;
        double supplied_m3=0,detached_m3=0;
    };

    void init(grid &film,std::uint32_t width,std::uint32_t height,
              vec2 metres_per_uv);
    void set_frame(grid &film,vec3 outward,mat4 transform);
    void set_chart(grid &film,const std::vector<chart_vertex> &vertices,
                   const std::vector<std::uint32_t> &indices);
    void deposit(grid &film,vec2 uv,float radius_m,float volume_m3);
    // Mass-conserving tangential gravity/capillary transport, porous uptake
    // and anisotropic wicking; this deliberately does not bake a stamp mask.
    void update(grid &film,float dt,const material &fluid,
                const porous::properties &substrate,
                vec3 gravity={0.f,-9.81f,0.f});
    float cell_area(const grid &film);
    double volume(const grid &film);
    // World-space release point for a future free drop.
    vec3 world_position(const grid &film,std::uint32_t x,std::uint32_t y);
    // Explicit mass transfer to 3D: subtract only AFTER emit succeeds.
    bool take(grid &film,std::uint32_t x,std::uint32_t y,float amount_m3);
}
