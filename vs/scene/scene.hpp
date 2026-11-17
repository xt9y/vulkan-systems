#pragma once
#include <vk/vk.hpp>
#include <vk/util/math/quaternion.hpp>
#include <vk/util/math/vector.hpp>
#include <array>
#include <cstdint>
#include <string>
#include <vector>
#include <span>
#include <unordered_map>
#include <memory>
#include <utility>
#include <iterator>
#include <stdexcept>
#include <cmath>

namespace vs::scene {
struct texture
{
    std::uint32_t id = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::vector<std::uint8_t> pixels;
    std::vector<std::uint8_t> encoded; // Original PNG/JPEG bytes for compact network transfer.
};

struct material
{
    std::array<float, 4> base_color{1, 1, 1, 1};
    std::array<float, 3> emissive{0, 0, 0};

    float metallic = 0;
    float roughness = 1;
    float occlusion = 1;
    float alpha_cutoff = 0.5f;

    std::uint32_t base_color_texture = 0;
    std::uint32_t metallic_roughness_texture = 0;
    std::uint32_t normal_texture = 0;
    std::uint32_t occlusion_texture = 0;
    std::uint32_t emissive_texture = 0;
    std::uint32_t alpha_mode = 0; // opaque, mask, blend
    bool double_sided = false;

    std::uint32_t procedural_pattern = 0;
};

struct vertex
{
    vk::util::math::vec3 position;
    vk::util::math::vec3 normal;

    float u = 0;
    float v = 0;

    std::array<std::uint32_t, 4> joints{};
    std::array<float, 4> weights{};
};

struct mesh
{
    std::uint32_t id = 0;
    std::uint32_t skin_id = 0;
    std::vector<vertex> vertices;
    std::vector<std::uint32_t> indices;

    material surface;
};

struct transform
{
    vk::util::math::vec3 position;
    vk::util::math::vec3 scale{1, 1, 1};
    vk::util::math::quat rotation;
};

// Shared model transform for physics, rays, GI and optical bounds.
inline vk::util::math::vec3 transform_position(const transform& pose,
                                               vk::util::math::vec3 point)
{
    return pose.position+vk::util::math::rotate(pose.rotation,{
        point.x*pose.scale.x,point.y*pose.scale.y,point.z*pose.scale.z});
}

struct clip
{
    std::string name;
    std::vector<std::uint8_t> bytes;
};

struct skin
{
    std::uint32_t id = 0;
    std::vector<std::uint8_t> skeleton;
    std::vector<clip> clips;
    std::vector<std::array<float, 16>> inverse_bind;
};

struct instance
{
    std::uint32_t id = 0;
    std::uint32_t mesh_id = 0;
    std::uint32_t animation = 0;
    std::uint32_t animation_tick = 0;
    std::uint32_t overlay_animation = UINT32_MAX;
    std::uint32_t overlay_root = UINT32_MAX;

    float overlay_weight = 0.0f;
    transform pose;
    vk::util::math::vec3 tint{1, 1, 1};
    std::uint32_t pattern = UINT32_MAX; // 0 solid, 1 checker, 2 glass, 3 reflective; UINT32_MAX inherits.
    std::uint32_t editor_group = 0; // Shared root instance ID for multi-mesh Mermaid models.
    float optical_strength = .9f; // transmission or reflectivity
    float optical_mattness = .05f; // [0,1]
    // Editor reference selects continuous Euler controls at/through the poles.
    // Rendering and simulation always consume pose.rotation.
    vk::util::math::vec3 rotation_reference{};
};

struct collision_box
{
    vk::util::math::vec3 minimum{};
    vk::util::math::vec3 maximum{};
};

enum class light_kind : std::uint32_t { plight=0, slight=1, lamp=2 };
struct point_light
{
    vk::util::math::vec3 position{};
    float radius = 1.0f;
    vk::util::math::vec3 color{1.0f, 1.0f, 1.0f};
    float strength = 1.0f;
    bool enabled = true;
    light_kind kind = light_kind::plight;
    vk::util::math::vec3 direction{0.0f,-1.0f,0.0f};
    float cone_degrees = 45.0f; // Full spotlight opening angle.
};

// VS defines only component/asset types, never a world or scene owner.
// Keep geometry ownership, persistence, IDs, and lifetime inside your project.
// Mesh IDs are shared across every instance; resolve them in one scan.
template<class Visit>
void each_mesh_instance(const std::vector<mesh>& meshes,
                        std::span<const instance> instances,Visit&& visit)
{
    std::unordered_map<std::uint32_t,const mesh*> by_id;
    by_id.reserve(meshes.size());
    for(const auto& m:meshes)by_id.emplace(m.id,&m);
    for(const auto& instance:instances)
        if(const auto found=by_id.find(instance.mesh_id);found!=by_id.end())
            visit(*found->second,instance);
}

// One camera convention: -Z forward, yaw/pitch/FOV in radians.
struct camera {
    vk::util::math::vec3 position{};
    float yaw=0.f,pitch=0.f,vertical_fov=1.04719755f;
};
struct camera_basis {vk::util::math::vec3 forward{},right{},up{};};
inline camera_basis axes(float yaw,float pitch) {
    const float sy=std::sin(yaw),cy=std::cos(yaw);
    const float sp=std::sin(pitch),cp=std::cos(pitch);
    return {{sy*cp,sp,-cy*cp},{cy,0.f,sy},{-sy*sp,cp,cy*sp}};
}
inline camera_basis axes(const camera& c) {return axes(c.yaw,c.pitch);}

// Canonical renderer-facing primitives. The application owns data instances;
// VS never owns or serializes a world and every renderer consumes this schema.
using vec3=vk::util::math::vec3;
struct surface {
    vec3 origin{},u{},v{},normal{};
    vec3 albedo{.725f,.710f,.680f},emission{};
    bool triangle=false; // Triangle: origin, origin+u, origin+v; otherwise quad.
    std::uint32_t optical_type=0; // 0 diffuse, 1 glass, 2 mirror
    float optical_strength=0.f,roughness=0.f,ior=1.52f;
    std::uint32_t pattern=0;
    // The same original per-face shading data also drives optical rays:
    // no separate normals/UV/texture arrays authored by application code.
    std::array<std::array<float,2>,3> uv{{{{0.f,0.f}},{{1.f,0.f}},{{0.f,1.f}}}};
    std::array<vec3,3> vertex_normals{};
    std::uint32_t base_color_texture=0,metallic_roughness_texture=0;
    std::uint32_t normal_texture=0,occlusion_texture=0,emissive_texture=0;
    float alpha_cutoff=0.f,base_alpha=1.f;
    // Generic geometry participation controls. Probes can sample a coarse
    // patch while shadow rays use separate exact triangles from the same scene.
    bool probe_receiver=true,shadow_caster=true;
};
struct triangle {vec3 a{},b{},c{};}; // Transient geometry, never a second scene.
struct cutout {std::uint32_t receiver=0,cover=0;};
struct glass_sphere {vec3 center{};float radius=0.f,ior=1.52f;};

// This is scene *data*, not a world/scene manager. No global state, registry,
// lifetime rules or entity behavior. Every project constructs and owns it.
struct data {
    std::vector<mesh> meshes;
    std::vector<instance> instances;
    std::vector<texture> textures;
    std::vector<skin> skins;
    std::vector<surface> surfaces;  // Procedural/static faces not stored as meshes.
    std::vector<point_light> lights;
    std::vector<collision_box> collision_boxes;
    std::vector<cutout> cutouts;
    glass_sphere glass{};
    std::uint32_t emitter=UINT32_MAX; // Optional explicit emissive face index.
    bool omnidirectional_emitter=false;
};

// One conversion path for mesh instances across all GI systems. Applies mesh
// material and instance optics without caching or duplicating renderer scenes.
template<class Visit>
void each_mesh_surface(const data& value, Visit&& visit) {
    each_mesh_instance(value.meshes,value.instances,[&](const mesh& m,const instance& entity) {
        for(std::size_t i=0;i+2<m.indices.size();i+=3) {
            const auto ia=m.indices[i],ib=m.indices[i+1],ic=m.indices[i+2];
            if(ia>=m.vertices.size()||ib>=m.vertices.size()||ic>=m.vertices.size())
                throw std::out_of_range("VS scene has an invalid mesh index");
            const auto a=transform_position(entity.pose,m.vertices[ia].position);
            const auto b=transform_position(entity.pose,m.vertices[ib].position);
            const auto c=transform_position(entity.pose,m.vertices[ic].position);
            const auto n=vk::util::math::cross(b-a,c-a);
            const auto length=vk::util::math::length(n);
            if(!(length>1.e-8f))continue;
            surface face{};
            face.origin=a;face.u=b-a;face.v=c-a;face.normal=n*(1.f/length);
            face.triangle=true;
            face.albedo={m.surface.base_color[0]*entity.tint.x,
                         m.surface.base_color[1]*entity.tint.y,
                         m.surface.base_color[2]*entity.tint.z};
            face.emission={m.surface.emissive[0],m.surface.emissive[1],m.surface.emissive[2]};
            face.roughness=m.surface.roughness;
            face.pattern=m.surface.procedural_pattern;
            face.optical_type=entity.pattern==2?1u:entity.pattern==3?2u:0u;
            face.optical_strength=face.optical_type?entity.optical_strength:0.f;
            const auto& va=m.vertices[ia];
            const auto& vb=m.vertices[ib];
            const auto& vc=m.vertices[ic];
            face.uv={{{{va.u,va.v}},{{vb.u,vb.v}},{{vc.u,vc.v}}}};
            const auto normal_at=[&](const vertex& vertex)->vec3 {
                const auto& scale=entity.pose.scale;
                const auto inverse=[](float v) {
                    return std::abs(v)>1.e-7f?1.f/v:1.f;
                };
                const vec3 n{
                    vertex.normal.x*inverse(scale.x),
                    vertex.normal.y*inverse(scale.y),
                    vertex.normal.z*inverse(scale.z)
                };
                const vec3 rotated=vk::util::math::rotate(entity.pose.rotation,n);
                const float length=vk::util::math::length(rotated);
                return length>1.e-7f?rotated*(1.f/length):face.normal;
            };
            face.vertex_normals={normal_at(va),normal_at(vb),normal_at(vc)};
            face.base_color_texture=m.surface.base_color_texture;
            face.metallic_roughness_texture=m.surface.metallic_roughness_texture;
            face.normal_texture=m.surface.normal_texture;
            face.occlusion_texture=m.surface.occlusion_texture;
            face.emissive_texture=m.surface.emissive_texture;
            face.alpha_cutoff=m.surface.alpha_cutoff;
            face.base_alpha=m.surface.base_color[3];
            visit(face);
        }
    });
}
template<class Visit>
void each_surface(const data& value,Visit&& visit) {
    for(const auto& face:value.surfaces)visit(face);
    each_mesh_surface(value,visit);
}
} // namespace scene
