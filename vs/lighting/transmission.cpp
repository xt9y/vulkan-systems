#include <vs/lighting/transmission.hpp>
#include <vs/bvh/stackless.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace vs::lighting {
namespace {
using v3=std::array<float,3>;
using v4=std::array<float,4>;
using node=::vs::bvh::stackless_node;

v3 vec(::vs::scene::vec3 x) {return {x.x,x.y,x.z};}
v3 add(v3 a,v3 b) {return {a[0]+b[0],a[1]+b[1],a[2]+b[2]};}
v3 sub(v3 a,v3 b) {return {a[0]-b[0],a[1]-b[1],a[2]-b[2]};}
v3 cross(v3 a,v3 b) {
    return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],
            a[0]*b[1]-a[1]*b[0]};
}
float dot(v3 a,v3 b) {return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
float sign(float x) {return x>=0.f?1.f:-1.f;}

v3 oct_direction(std::uint32_t x,std::uint32_t y) {
    const float scale=2.f/float(transmission_resolution);
    float a=(float(x)+.5f)*scale-1.f;
    float b=(float(y)+.5f)*scale-1.f;
    const float z=1.f-std::abs(a)-std::abs(b);
    if(z<0.f) {
        const float first=(1.f-std::abs(b))*sign(a);
        b=(1.f-std::abs(a))*sign(b);
        a=first;
    }
    const float length=std::sqrt(a*a+b*b+z*z);
    return {a/length,b/length,z/length};
}

struct material {v3 filter{};};
struct geometry {
    std::vector<v4> triangles;
    std::vector<node> nodes;
    std::vector<material> materials;
    std::vector<std::array<v3,2>> bounds;

    void add_triangle(v3 a,v3 b,v3 c,std::size_t index) {
        const auto n=cross(sub(b,a),sub(c,a));
        if(dot(n,n)<=1e-14f)return;
        if(index>=16777216u)
            throw std::length_error("Glass material identifier exceeds float precision");
        triangles.push_back({a[0],a[1],a[2],float(index)});
        triangles.push_back({b[0],b[1],b[2],0.f});
        triangles.push_back({c[0],c[1],c[2],0.f});
    }
};

bool intersects_sphere(const std::array<v3,2>& bounds,v3 center,float radius) {
    float squared=0.f;
    for(int k=0;k<3;++k) {
        const float d=center[k]-std::clamp(center[k],bounds[0][k],bounds[1][k]);
        squared+=d*d;
    }
    return squared<=radius*radius;
}
bool hit_box(v3 eye,v3 ray,float far,const node& box) {
    float near=0.f;
    for(int i=0;i<3;++i) {
        if(std::abs(ray[i])<1e-8f) {
            if(eye[i]<box.minimum[i] || eye[i]>box.maximum[i])return false;
            continue;
        }
        float a=(box.minimum[i]-eye[i])/ray[i];
        float b=(box.maximum[i]-eye[i])/ray[i];
        if(a>b)std::swap(a,b);
        near=std::max(near,a);
        far=std::min(far,b);
        if(near>far)return false;
    }
    return true;
}
float hit_triangle(v3 eye,v3 ray,float far,
                   const std::vector<v4>& triangles,std::size_t i) {
    const auto& p=triangles[i],&q=triangles[i+1],&r=triangles[i+2];
    const v3 a{p[0],p[1],p[2]};
    const v3 u{q[0],q[1],q[2]},v{r[0],r[1],r[2]};
    const v3 h=cross(ray,v);
    const float det=dot(u,h);
    if(std::abs(det)<1e-7f)return far;
    const float inverse=1.f/det;
    const v3 t=sub(eye,a);
    const float x=dot(t,h)*inverse;
    if(x<0.f || x>1.f)return far;
    const v3 b=cross(t,u);
    const float y=dot(ray,b)*inverse;
    if(y<0.f || x+y>1.f)return far;
    const float distance=dot(v,b)*inverse;
    return distance>0.001f && distance<far?distance:far;
}
struct crossing {
    float distance=std::numeric_limits<float>::infinity();
    std::uint32_t material=0;
};

std::array<crossing,transmission_layers> trace(
    const geometry& glass,v3 eye,v3 direction,float far) {
    std::array<crossing,transmission_layers> closest{};
    std::uint32_t index=0;
    while(index<glass.nodes.size()) {
        const auto& box=glass.nodes[index];
        if(!hit_box(eye,direction,far,box)) {
            index=std::max(box.next,index+1);
            continue;
        }
        const std::uint32_t count=box.first_and_count&15u;
        const std::uint32_t first=box.first_and_count>>4u;
        if(count)for(std::uint32_t j=0;j<count;++j) {
            const std::size_t base=std::size_t(first+j)*3u;
            const float t=hit_triangle(eye,direction,far,glass.triangles,base);
            if(t>=far || t>=closest.back().distance)continue;
            const auto id=std::uint32_t(glass.triangles[base][3]);
            // Shared edges must not apply the same glass boundary twice.
            bool duplicate=false;
            for(const auto& item:closest)
                if(item.material==id && std::abs(item.distance-t)<.001f)
                    duplicate=true;
            if(duplicate)continue;
            std::size_t position=0;
            while(position<closest.size() && closest[position].distance<t)
                ++position;
            if(position==closest.size())continue;
            for(std::size_t k=closest.size()-1;k>position;--k)
                closest[k]=closest[k-1];
            closest[position]={t,id};
        }
        ++index;
    }
    return closest;
}
std::uint32_t pack(v3 filter,float distance,float radius) {
    const auto channel=[](float v) ->std::uint32_t {
        return std::uint32_t(std::lround(std::clamp(v,0.f,1.f)*255.f));
    };
    const auto depth=std::uint32_t(std::clamp(
        std::lround(distance/radius*254.f),0l,254l));
    return channel(filter[0]) | (channel(filter[1])<<8u) |
        (channel(filter[2])<<16u) | (depth<<24u);
}
} // namespace

transmission_atlas bake_transmission(
    std::span<const ::vs::scene::surface> surfaces,
    std::span<const ::vs::scene::point_light> lights) {
    if(lights.size()>transmission_max_lights)
        throw std::invalid_argument("Transmission supports at most 64 point lights");
    transmission_atlas result{};
    result.words.resize(transmission_header_words,0u);
    geometry glass{};
    for(const auto& face:surfaces) {
        if(face.optical_type!=1)continue;
        const auto a=vec(face.origin),b=add(a,vec(face.u)),
                   c=add(a,vec(face.v));
        const auto fourth=add(b,vec(face.v));
        const std::size_t before=glass.triangles.size();
        const auto index=glass.materials.size();
        glass.add_triangle(a,b,c,index);
        if(!face.triangle)glass.add_triangle(b,fourth,c,index);
        if(before==glass.triangles.size())continue;
        const auto pigment=vec(face.albedo);
        const float strength=std::clamp(face.optical_strength,0.f,1.f);
        const float ior=std::max(1.f,face.ior);
        const float f0=std::pow((ior-1.f)/(ior+1.f),2.f);
        material m{};
        for(int k=0;k<3;++k) {
            const float tint=std::clamp(pigment[k],1.f/255.f,1.f);
            // optical_strength is *pass-through*, not dye density. A pane
            // at zero transmission blocks light rather than disappearing
            // from the lighting BVH. Fresnel is paid at each boundary.
            m.filter[k]=strength*std::pow(tint,1.f)*(1.f-f0);
        }
        glass.materials.push_back(m);
        std::array<v3,2> bounds{a,a};
        const auto expand=[&](v3 p) {
            for(int k=0;k<3;++k) {
                bounds[0][k]=std::min(bounds[0][k],p[k]);
                bounds[1][k]=std::max(bounds[1][k],p[k]);
            }
        };
        expand(b);expand(c);if(!face.triangle)expand(fourth);
        glass.bounds.push_back(bounds);
    }
    result.glass_triangles=std::uint32_t(glass.triangles.size()/3);
    if(glass.triangles.empty())return result;
    glass.nodes=::vs::bvh::build_stackless(glass.triangles);
    ::vs::bvh::store_triangle_edges(glass.triangles);
    constexpr std::size_t words_per_light=std::size_t(transmission_resolution)*
        transmission_resolution*transmission_layers;
    // Only affected lights consume map storage. Flags hold 1-based map
    // indices so the shader avoids a sparse 64-light allocation.
    for(std::size_t i=0;i<lights.size();++i) {
        const auto& light=lights[i];
        if(!light.enabled || !(light.strength>0.f) || !(light.radius>0.f))
            continue;
        const v3 eye=vec(light.position);
        bool relevant=false;
        for(const auto& bounds:glass.bounds)
            if(intersects_sphere(bounds,eye,light.radius)) {
                relevant=true;
                break;
            }
        if(!relevant)continue;
        const auto map=result.active_lights++;
        result.words[i]=map+1u;
        const auto start=result.words.size();
        result.words.resize(start+words_per_light,0xffffffffu);
        for(std::uint32_t y=0;y<transmission_resolution;++y)
        for(std::uint32_t x=0;x<transmission_resolution;++x) {
            const auto intersections=trace(glass,eye,oct_direction(x,y),light.radius);
            const auto first=start+
                (std::size_t(y)*transmission_resolution+x)*transmission_layers;
            for(std::uint32_t layer=0;layer<transmission_layers;++layer) {
                const auto& crossing=intersections[layer];
                if(!std::isfinite(crossing.distance))break;
                result.words[first+layer]=pack(
                    glass.materials[crossing.material].filter,crossing.distance,
                    light.radius);
            }
        }
    }
    result.nodes=std::move(glass.nodes);
    result.triangles=std::move(glass.triangles);
    result.filters.reserve(glass.materials.size());
    for(const auto& material:glass.materials)
        result.filters.push_back({
            material.filter[0],material.filter[1],material.filter[2],0.f});
    return result;
}
transmission_atlas bake_transmission(const ::vs::scene::data& value) {
    std::vector<::vs::scene::surface> faces=value.surfaces;
    ::vs::scene::each_mesh_surface(value,[&](const auto& face) {
        if(face.optical_type==1u)faces.push_back(face);
    });
    return bake_transmission(faces,value.lights);
}
} // namespace vs::lighting
