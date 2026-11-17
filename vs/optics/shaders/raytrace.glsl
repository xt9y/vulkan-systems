// Reusable software-BVH reflection queries over Photon Mapping's scene
// buffers. BVH leaf ordering and the 112-byte face layout match VulkanPMap.
// Full-resolution reflected rays use the same triangle BVH as Photon
// Mapping. Per-hit shadows, photon irradiance, UV texture, and smooth
// vertex normals are evaluated without an interpolated radiance cache.
#ifndef OPTICS_RAYTRACE_GLSL
#define OPTICS_RAYTRACE_GLSL
#include "lighting_shared.glsl"
const float OPTICS_PI=3.14159265358979;
struct OpticsFace {vec4 origin,u,v,normal,albedo,emission,metric;};
struct OpticsNode {vec3 minimum;uint next;vec3 maximum;uint first_and_count;};
struct OpticsPhoton {vec4 position,power;uvec4 link;};
struct OpticsLight {vec4 position_radius,color_strength;};
layout(std430,set=0,binding=6) readonly buffer OpticalFaces {OpticsFace optics_faces[];};
layout(std430,set=0,binding=7) readonly buffer OpticalNodes {OpticsNode optics_nodes[];};
layout(std430,set=0,binding=8) readonly buffer OpticalFaceIds {uint optics_face_ids[];};
layout(std430,set=0,binding=9) readonly buffer OpticalScene {
    uvec4 optics_desc;
    vec4 optics_grid;
    uvec4 optics_extra;
};
layout(std430,set=0,binding=10) readonly buffer OpticalPhotons {OpticsPhoton optics_photons[];};
layout(std430,set=0,binding=11) readonly buffer OpticalOffsets {uint optics_offsets[];};
layout(std430,set=0,binding=12) readonly buffer OpticalLights {
    uvec4 optics_light_count;
    OpticsLight optics_lights[16];
};
// A face maps its three source UVs onto a subrectangle of the packed
// RGBA8 atlas. Face indices match the Photon Mapping BVH exactly.
struct OpticsUV {
    vec4 uv01,uv2_origin,rect;
    vec4 metallic_rect,normal_rect,occlusion_rect,emissive_rect;
};
layout(std430,set=0,binding=13) readonly buffer OpticalUV {
    OpticsUV optics_uv[];
};
layout(set=0,binding=14) uniform sampler2D optics_atlas;
// Per-face vertex normals share the exact face ordering of the traced BVH.
// They do not change triangle intersections; curved meshes keep smooth
// reflected directions instead of showing their individual polygons.
struct OpticsNormals {vec4 a,b,c;};
layout(std430,set=0,binding=15) readonly buffer OpticalNormals {
    OpticsNormals optics_normals[];
};
layout(set=0,binding=16) uniform sampler3D optics_probes;
layout(std430,set=0,binding=17) readonly buffer OpticalProbeSpace {
    vec4 optics_probe_space;
};
layout(set=0,binding=18) uniform sampler2D hdr_image;
struct OpticsHit {
    vec3 position,normal,geometric_normal;
    float t;
    uint face_id;
    vec2 bary;
};
// Preserve the exact slab divisions while computing near-zero direction
// handling once per ray rather than for every BVH node.
vec3 optics_safe_direction(vec3 rd)
{
    return vec3(
        abs(rd.x)<1e-8?(rd.x<0.0?-1e-8:1e-8):rd.x,
        abs(rd.y)<1e-8?(rd.y<0.0?-1e-8:1e-8):rd.y,
        abs(rd.z)<1e-8?(rd.z<0.0?-1e-8:1e-8):rd.z);
}
bool optics_aabb(vec3 ro,vec3 inv_direction,vec3 lo,vec3 hi,float max_t)
{
    // One reciprocal per ray, not six divisions for every BVH node.
    // The slab endpoints and acceptance bounds are unchanged.
    vec3 lo_t=(lo-ro)*inv_direction;
    vec3 hi_t=(hi-ro)*inv_direction;
    vec3 minimum=min(lo_t,hi_t),maximum=max(lo_t,hi_t);
    float near_t=max(max(minimum.x,minimum.y),max(minimum.z,.025));
    float far_t=min(min(maximum.x,maximum.y),min(maximum.z,max_t));
    return near_t<=far_t;
}
vec2 optics_atlas_coord(OpticsUV mapping,vec2 bary,vec4 area)
{
    vec2 uv=fract(mapping.uv01.xy*(1.0-bary.x-bary.y)+
                  mapping.uv01.zw*bary.x+
                  mapping.uv2_origin.xy*bary.y);
    vec2 size=vec2(textureSize(optics_atlas,0));
    vec2 lo=area.xy+.5/size;
    vec2 hi=area.xy+area.zw-.5/size;
    return mix(lo,hi,uv);
}
vec2 optics_atlas_coord(OpticsUV mapping,vec2 bary)
{
    return optics_atlas_coord(mapping,bary,
        vec4(mapping.uv2_origin.zw,mapping.rect.xy));
}
bool optics_face_hit(vec3 ro,vec3 rd,OpticsFace f,uint face_id,
                     float limit,out float t)
{
    float dot_n=dot(rd,f.normal.xyz);
    if(abs(dot_n)<1e-6)return false;
    float dist=dot(f.origin.xyz-ro,f.normal.xyz)/dot_n;
    if(dist<=.025 || dist>=limit)return false;
    vec3 d=ro+rd*dist-f.origin.xyz;
    vec2 q=vec2(dot(d,f.u.xyz),dot(d,f.v.xyz));
    float u=q.x*f.metric.x+q.y*f.metric.y;
    float v=q.x*f.metric.y+q.y*f.metric.z;
    if(u<-.0001 || v<-.0001 || u>1.0001 || v>1.0001 ||
       (f.normal.w>.5 && u+v>1.0001))return false;
    OpticsUV mapping=optics_uv[face_id];
    if(mapping.rect.w>0.0) {
        // Match the raster alpha test for textured *and* untextured
        // Mermaid cards. Previously untextured masked faces never clipped,
        // producing ghost geometry that hid correctly textured surfaces.
        float alpha=mapping.rect.z;
        if(mapping.rect.x>0.0)
            alpha*=textureLod(optics_atlas,
                optics_atlas_coord(mapping,vec2(u,v)),0.0).a;
        if(alpha<mapping.rect.w)return false;
    }
    t=dist;
    return true;
}
bool optics_closest(vec3 ro,vec3 rd,float max_t,out OpticsHit result)
{
    if(optics_desc.x==0u)return false;
    uint total_nodes=optics_nodes[0].next;
    if(total_nodes==0u || total_nodes>1000000u)return false;
    result.t=max_t;
    bool found=false;
    uint index=0u;
    vec3 inv_direction=1.0/optics_safe_direction(rd);
    for(uint steps=0u;index<total_nodes && steps<total_nodes*2u;++steps) {
        OpticsNode node=optics_nodes[index];
        if(!optics_aabb(ro,inv_direction,node.minimum,node.maximum,result.t)) {
            index=node.next;
            continue;
        }
        uint count=node.first_and_count&15u;
        if(count==0u) {++index;continue;}
        uint first=node.first_and_count>>4u;
        for(uint k=0u;k<count;++k) {
            uint face_id=optics_face_ids[first+k];
            // The light emitter is only a transport source. It must not
            // appear as a white rectangular mesh in camera/reflection rays.
            if(face_id>=optics_desc.x || face_id==optics_desc.y)continue;
            float t;
            if(optics_face_hit(ro,rd,optics_faces[face_id],face_id,result.t,t)) {
                result.t=t;
                result.face_id=face_id;
                found=true;
            }
        }
        index=node.next;
    }
    if(found) {
        OpticsFace f=optics_faces[result.face_id];
        result.position=ro+rd*result.t;
        vec3 d=result.position-f.origin.xyz;
        vec2 q=vec2(dot(d,f.u.xyz),dot(d,f.v.xyz));
        result.bary=vec2(q.x*f.metric.x+q.y*f.metric.y,
                         q.x*f.metric.y+q.y*f.metric.z);
        OpticsNormals hit_normals=optics_normals[result.face_id];
        vec3 n=hit_normals.a.xyz*(1.0-result.bary.x-result.bary.y)+
               hit_normals.b.xyz*result.bary.x+hit_normals.c.xyz*result.bary.y;
        vec3 geometric=normalize(f.normal.xyz);
        if(dot(n,n)<1e-8)n=geometric;
        else n=normalize(n);
        // Keep shading normals in the same hemisphere as the triangle.
        if(dot(n,geometric)<0.0)n=-n;
        if(dot(rd,geometric)>0.0) {
            geometric=-geometric;
            n=-n;
        }
        result.geometric_normal=geometric;
        result.normal=n;
    }
    return found;
}
// Direct point-light shadow visibility needs only an opaque intersection,
// not a nearest surface with all the associated memory traffic. Transparent
// glass does not block ordinary lights; reflective and opaque surfaces do.
bool optics_occluded(vec3 ro,vec3 rd,float limit,uint source)
{
    if(limit<=.025 || optics_desc.x==0u)return false;
    uint total=optics_nodes[0].next;
    if(total==0u || total>1000000u)return false;
    uint index=0u;
    vec3 inv_direction=1.0/optics_safe_direction(rd);
    for(uint step=0u;index<total && step<total*2u;++step) {
        OpticsNode node=optics_nodes[index];
        if(!optics_aabb(ro,inv_direction,node.minimum,node.maximum,limit)) {
            index=node.next;
            continue;
        }
        uint count=node.first_and_count&15u;
        if(count==0u){++index;continue;}
        uint first=node.first_and_count>>4u;
        for(uint k=0u;k<count;++k) {
            uint id=optics_face_ids[first+k];
            if(id>=optics_desc.x || id==optics_desc.y || id==source)continue;
            OpticsFace f=optics_faces[id];
            if(f.albedo.w>.001)continue;
            float t;
            if(optics_face_hit(ro,rd,f,id,limit,t))return true;
        }
        index=node.next;
    }
    return false;
}
vec3 optics_photon_radiance(OpticsHit hit,vec3 albedo)
{
    // Use the existing photon grid for off-screen indirect radiance,
    // limiting work per query so the first mirror stays affordable on M2.
    float radius=max(.18,optics_grid.w);
    float radius2=radius*radius;
    ivec3 center=ivec3(floor((hit.position-optics_grid.xyz)/
                             max(optics_grid.w,1e-5)));
    vec3 flux=vec3(0);
    for(int z=-1;z<=1;++z)
    for(int y=-1;y<=1;++y)
    for(int x=-1;x<=1;++x) {
        ivec3 c=center+ivec3(x,y,z);
        if(any(lessThan(c,ivec3(0))) ||
           any(greaterThanEqual(c,ivec3(16))))continue;
        uint cell=uint(c.x+16*(c.y+16*c.z));
        uint start=optics_offsets[cell];
        uint stop=min(optics_offsets[cell+1u],start+48u);
        for(uint i=start;i<stop;++i) {
            OpticsPhoton p=optics_photons[i];
            if(p.link.y!=hit.face_id)continue;
            vec3 delta=hit.position-p.position.xyz;
            float distance2=dot(delta,delta);
            if(distance2>=radius2)continue;
            flux+=p.power.rgb*(1.0-distance2/radius2);
        }
    }
    return flux*albedo*(2.0/(OPTICS_PI*OPTICS_PI*radius2));
}
vec3 optics_surface_radiance(OpticsHit hit,vec3 rd)
{
    OpticsFace f=optics_faces[hit.face_id];
    OpticsUV mapping=optics_uv[hit.face_id];
    OpticsNormals material=optics_normals[hit.face_id];
    vec3 albedo=max(f.albedo.rgb,vec3(0.0));
    vec2 bary=hit.bary;
    float metallic=clamp(material.a.w,0.0,1.0);
    float roughness=clamp(material.b.w,.045,1.0);
    float ao=clamp(material.c.w,0.0,1.0);
    if(f.emission.w>.1 && f.emission.w<.75) {
        if(f.emission.w<.375) {
            ivec2 tile=ivec2(floor(hit.position.xz*2.0));
            albedo=((tile.x+tile.y)&1)!=0 ?
                vec3(1.0,0.0,1.0):vec3(0.0);
        } else {
            vec2 texcoord=mapping.uv01.xy*(1.0-bary.x-bary.y)+
                mapping.uv01.zw*bary.x+mapping.uv2_origin.xy*bary.y;
            ivec2 tile=ivec2(floor(texcoord*4.0));
            albedo*=((tile.x+tile.y)&1)!=0?1.0:.08;
        }
    }
    if(mapping.rect.x>0.0)
        albedo*=pow(max(textureLod(optics_atlas,
            optics_atlas_coord(mapping,bary),0.0).rgb,vec3(0.0)),
            vec3(2.2));
    if(mapping.metallic_rect.z>0.0) {
        vec4 mr=textureLod(optics_atlas,
            optics_atlas_coord(mapping,bary,mapping.metallic_rect),0.0);
        roughness*=mr.g;
        metallic*=mr.b;
    }
    roughness=clamp(roughness,.045,1.0);
    vec3 N=hit.normal;
    if(mapping.normal_rect.z>0.0) {
        vec2 du=mapping.uv01.zw-mapping.uv01.xy;
        vec2 dv=mapping.uv2_origin.xy-mapping.uv01.xy;
        float det=du.x*dv.y-du.y*dv.x;
        if(abs(det)>1e-7) {
            vec3 T=normalize((f.u.xyz*dv.y-f.v.xyz*du.y)/det);
            vec3 B=normalize((f.v.xyz*du.x-f.u.xyz*dv.x)/det);
            vec3 tex_normal=textureLod(optics_atlas,
                optics_atlas_coord(mapping,bary,mapping.normal_rect),
                0.0).xyz*2.0-1.0;
            N=normalize(mat3(T,B,N)*tex_normal);
            if(dot(N,hit.geometric_normal)<0.0)N=-N;
        }
    }
    if(mapping.occlusion_rect.z>0.0)
        ao*=textureLod(optics_atlas,
            optics_atlas_coord(mapping,bary,mapping.occlusion_rect),0.0).r;
    vec3 emission=max(f.emission.rgb,vec3(0.0));
    if(mapping.emissive_rect.z>0.0)
        emission*=pow(max(textureLod(optics_atlas,
            optics_atlas_coord(mapping,bary,mapping.emissive_rect),
            0.0).rgb,vec3(0.0)),vec3(2.2));
    vec3 N_probe=N;
    vec3 bounce=optics_probe_space.w>0.0 ?
        shared_indirect_lighting(optics_probes,optics_probe_space.xyz,
                                 optics_probe_space.w,hit.position,N_probe)*
        (.01*min(1.0,roughness*1.5)) :
        optics_photon_radiance(hit,vec3(1.0));
    vec3 radiance=emission+albedo*(1.0-metallic)*ao*bounce;
    vec3 V=normalize(-rd);
    for(uint i=0u;i<min(optics_light_count.x,16u);++i) {
        OpticsLight light=optics_lights[i];
        vec3 to_light=light.position_radius.xyz-hit.position;
        float radius=max(light.position_radius.w,.001);
        float dist2=dot(to_light,to_light);
        if(dist2<.000004 || dist2>=radius*radius ||
           dot(N,to_light)<=0.0)continue;
        float distance=sqrt(dist2);
        vec3 L=to_light/distance;
        if(optics_occluded(hit.position+N*.004,L,
                           distance-.008,hit.face_id))continue;
        float q=clamp(distance/radius,0.0,1.0);
        float attenuation=1.0-q*q*(3.0-2.0*q);
        radiance+=shared_brdf(albedo,metallic,roughness,N,V,L)*
            light.color_strength.rgb*
            (light.color_strength.w*attenuation);
    }
    return radiance;
}
#ifndef OPTICS_ONLY_MIRRORS
struct OpticsPath {
    vec3 origin;
    vec3 direction;
    vec3 throughput;
    int depth;
};
vec3 optics_offscreen_radiance(vec3 origin,vec3 direction)
{
    if(optics_desc.x==0u)return vec3(0.0);
    OpticsPath pending[16];
    int queued=1;
    pending[0]=OpticsPath(origin,normalize(direction),vec3(1.0),0);
    vec3 accumulated=vec3(0.0);
    // Material interactions are bounded, not recursively compiled. Eight
    // boundaries permit glass -> glass -> mirror -> glass and exits.
    for(int iteration=0;iteration<64 && queued>0;++iteration) {
        OpticsPath path=pending[--queued];
        if(path.depth>=10 ||
           max(path.throughput.r,max(path.throughput.g,
               path.throughput.b))<.003)continue;
        OpticsHit hit;
        if(!optics_closest(path.origin,path.direction,1000.0,hit))
            continue; // Outside the real scene: no cubemap/sky substitute.
        OpticsFace surface=optics_faces[hit.face_id];
        if(surface.emission.w>1.0) {
            float strength=clamp(surface.albedo.w,0.0,1.0);
            float ior=max(1.001,surface.emission.w);
            bool entering=dot(path.direction,surface.normal.xyz)<0.0;
            vec3 normal=hit.normal;
            float cosine=clamp(dot(-path.direction,normal),0.0,1.0);
            float r0=(ior-1.0)/(ior+1.0);
            r0*=r0;
            float fresnel=r0+(1.0-r0)*pow(1.0-cosine,5.0);
            // Thin-window material: no angular refraction and no lensing.
            // The direction remains the same at every material boundary.
            vec3 tint=max(surface.albedo.rgb,vec3(0.0));
            // Low pass-through still reveals the material itself and
            // keeps the edge visible in nested reflections/refractions.
            // An exiting surface does not bend or tint the ray a second time.
            // This preserves the perspective of the original camera and
            // avoids double surfaces on thick glass cubes.
            if(!entering) {
                if(queued<16)
                    pending[queued++]=OpticsPath(
                        hit.position+path.direction*.03,path.direction,
                        path.throughput,path.depth+1);
                continue;
            }
            if(entering)
                accumulated+=path.throughput*tint*(1.0-strength)*
                             (1.0-fresnel);
            vec3 reflected=path.throughput*fresnel;
            if(queued<16 && max(reflected.r,max(reflected.g,reflected.b))>.003)
                pending[queued++]=OpticsPath(
                    hit.position+hit.geometric_normal*.012,
                    normalize(reflect(path.direction,normal)),
                    reflected,path.depth+1);
            if(strength>.003) {
                vec3 transmitted=path.throughput*(1.0-fresnel)*strength*tint;
                if(queued<16)
                    pending[queued++]=OpticsPath(
                        hit.position+path.direction*.03,path.direction,
                        transmitted,path.depth+1);
            }
            continue;
        }
        if(surface.emission.w<-.001) {
            float reflectivity=clamp(-surface.emission.w,0.0,1.0);
            // Mirror reflectivity remains full strength, but the material
            // color acts as a per-channel filter at each nested bounce.
            path.throughput*=clamp(surface.albedo.rgb,vec3(0.0),vec3(1.0));
            path.origin=hit.position+hit.geometric_normal*.012;
            path.direction=normalize(reflect(path.direction,hit.normal));
            path.depth++;
            if(queued<16)pending[queued++]=path;
            continue;
        }
        // Always shade the traced surface with its own triangle UVs.
        // Switching between a screen-space HDR lookup and the atlas when a
        // hit crosses the FOV/depth threshold changes textures abruptly:
        // transparent Mermaid cards can flicker or disappear entirely.
        accumulated+=path.throughput*
                     optics_surface_radiance(hit,path.direction);
    }
    return accumulated;
}
#endif
// Hot path for 100% mirrors: no Fresnel branches, queue, or per-bounce
// throughput state when the reflected scene contains only opaque surfaces
// and other perfect mirrors. Exact same closest-hit, normals and shading.
// If a glass boundary is reached, replay with the original general tracer
// so nested dielectric reflections keep the same depth/energy accounting.
vec3 optics_perfect_mirror(vec3 origin,vec3 direction)
{
    vec3 start_origin=origin;
    vec3 start_direction=direction;
    vec3 throughput=vec3(1.0);
    vec3 ray=normalize(direction);
    for(int depth=0;depth<10;++depth) {
        OpticsHit hit;
        if(!optics_closest(origin,ray,1000.0,hit))return vec3(0.0);
        OpticsFace surface=optics_faces[hit.face_id];
        if(surface.emission.w>1.0) {
#ifndef OPTICS_ONLY_MIRRORS
            return optics_offscreen_radiance(start_origin,start_direction);
#else
            // The mirror-only variant is selected only when the authored
            // scene contains no dielectric faces; this is unreachable.
            return vec3(0.0);
#endif
        }
        if(surface.emission.w<-.001) {
            throughput*=clamp(surface.albedo.rgb,vec3(0.0),vec3(1.0));
            origin=hit.position+hit.geometric_normal*.012;
            ray=normalize(reflect(ray,hit.normal));
            continue;
        }
        return throughput*optics_surface_radiance(hit,ray);
    }
    return vec3(0.0);
}
#ifndef OPTICS_ONLY_MIRRORS
vec3 optics_offscreen_glossy(vec3 origin,vec3 direction,float mattness)
{
    vec3 center=optics_offscreen_radiance(origin,direction);
    // Deterministic cone sampling only for matte, off-screen reflections.
    // No noise, stochastic temporal filter or environment fallback.
    if(mattness<=.35)return center;
    vec3 axis=abs(direction.y)<.9?vec3(0,1,0):vec3(1,0,0);
    vec3 tangent=normalize(cross(direction,axis));
    vec3 bitangent=normalize(cross(direction,tangent));
    float spread=mattness*mattness*.16;
    vec3 offset=(tangent+bitangent)*spread;
    vec3 a=optics_offscreen_radiance(origin,normalize(direction+offset));
    vec3 b=optics_offscreen_radiance(origin,normalize(direction-offset));
    return (center*2.0+a+b)*.25;
}
#endif
#endif
