// Reusable software-BVH reflection queries over Photon Mapping's scene
// buffers. BVH leaf ordering and the 112-byte face layout match VulkanPMap.
// Queries only run for material pixels whose reflected/refracted scene data
// is unavailable in the visible rasterized image.
#ifndef OPTICS_RAYTRACE_GLSL
#define OPTICS_RAYTRACE_GLSL
const float OPTICS_PI=3.14159265358979;
struct OpticsFace {vec4 origin,u,v,normal,albedo,emission,metric;};
struct OpticsNode {vec3 minimum;uint next;vec3 maximum;uint first_and_count;};
struct OpticsPhoton {vec4 position,power;uvec4 link;};
struct OpticsLight {vec4 position_radius,color_strength;};
layout(std430,set=0,binding=6) readonly buffer OpticalFaces {OpticsFace optics_faces[];};
layout(std430,set=0,binding=7) readonly buffer OpticalNodes {OpticsNode optics_nodes[];};
layout(std430,set=0,binding=8) readonly buffer OpticalTriangles {vec4 optics_triangles[];};
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
struct OpticsHit {vec3 position,normal;float t;uint face_id;};
bool optics_aabb(vec3 ro,vec3 rd,vec3 lo,vec3 hi,float max_t)
{
    vec3 safe=vec3(
        abs(rd.x)<1e-8?(rd.x<0.0?-1e-8:1e-8):rd.x,
        abs(rd.y)<1e-8?(rd.y<0.0?-1e-8:1e-8):rd.y,
        abs(rd.z)<1e-8?(rd.z<0.0?-1e-8:1e-8):rd.z);
    vec3 lo_t=(lo-ro)/safe;
    vec3 hi_t=(hi-ro)/safe;
    vec3 minimum=min(lo_t,hi_t),maximum=max(lo_t,hi_t);
    float near_t=max(max(minimum.x,minimum.y),max(minimum.z,.025));
    float far_t=min(min(maximum.x,maximum.y),min(maximum.z,max_t));
    return near_t<=far_t;
}
bool optics_face_hit(vec3 ro,vec3 rd,OpticsFace f,float limit,out float t)
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
    for(uint steps=0u;index<total_nodes && steps<total_nodes*2u;++steps) {
        OpticsNode node=optics_nodes[index];
        if(!optics_aabb(ro,rd,node.minimum,node.maximum,result.t)) {
            index=node.next;
            continue;
        }
        uint count=node.first_and_count&15u;
        if(count==0u) {++index;continue;}
        uint first=node.first_and_count>>4u;
        for(uint k=0u;k<count;++k) {
            uint face_id=uint(optics_triangles[(first+k)*3u].w);
            if(face_id>=optics_desc.x)continue;
            float t;
            if(optics_face_hit(ro,rd,optics_faces[face_id],result.t,t)) {
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
        result.normal=dot(rd,f.normal.xyz)<0.0?f.normal.xyz:-f.normal.xyz;
    }
    return found;
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
    vec3 albedo=max(f.albedo.rgb,vec3(0.0));
    if(f.emission.w>.1 && f.emission.w<.75) {
        if(f.emission.w<.375) {
            // Same 0.5-metre world-space magenta checker as shd/mesh.frag.
            ivec2 tile=ivec2(floor(hit.position.xz*2.0));
            albedo=((tile.x+tile.y)&1)!=0 ?
                   vec3(1.0,0.0,1.0):vec3(0.0);
        } else {
            // Cube editor's four local squares across each material face.
            vec3 d=hit.position-f.origin.xyz;
            vec2 q=vec2(dot(d,f.u.xyz),dot(d,f.v.xyz));
            vec2 uv=vec2(q.x*f.metric.x+q.y*f.metric.y,
                          q.x*f.metric.y+q.y*f.metric.z);
            ivec2 tile=ivec2(floor(uv*4.0));
            albedo*=((tile.x+tile.y)&1)!=0?1.0:.08;
        }
    }
    vec3 radiance=max(f.emission.rgb,vec3(0.0))+
                  optics_photon_radiance(hit,albedo);
    vec3 normal=hit.normal;
    for(uint i=0u;i<min(optics_light_count.x,16u);++i) {
        OpticsLight light=optics_lights[i];
        vec3 delta=light.position_radius.xyz-hit.position;
        float dist2=dot(delta,delta);
        float radius=max(light.position_radius.w,.001);
        if(dist2>=radius*radius || dist2<.0001)continue;
        float distance=sqrt(dist2);
        vec3 L=delta/distance;
        float lambert=max(dot(normal,L),0.0);
        if(lambert<=0.0)continue;
        // The reflected view must preserve actual hard shadowing.
        OpticsHit blocker;
        bool occluded=optics_closest(hit.position+normal*.03,L,
                                    distance-.06,blocker);
        if(occluded && optics_faces[blocker.face_id].albedo.w<=.001)
            continue;
        float fade=max(0.0,1.0-distance/radius);
        radiance+=albedo*light.color_strength.rgb*
            (light.color_strength.w*lambert*fade*fade/
             (1.0+.12*dist2));
    }
    return radiance;
}
vec3 optics_offscreen_radiance(vec3 ro,vec3 direction)
{
    if(optics_desc.x==0u)return vec3(0.0);
    vec3 energy=vec3(1);
    for(int bounce=0;bounce<3;++bounce) {
        OpticsHit hit;
        if(!optics_closest(ro,direction,1000.0,hit))return vec3(0.0);
        OpticsFace f=optics_faces[hit.face_id];
        if(f.albedo.w>.001) {
            float ior=max(1.001,f.emission.w);
            float eta=dot(direction,f.normal.xyz)<0.0?1.0/ior:ior;
            vec3 n=hit.normal;
            vec3 refracted=refract(direction,n,eta);
            direction=dot(refracted,refracted)>.0001?
                normalize(refracted):reflect(direction,n);
            energy*=max(f.albedo.rgb,vec3(0.0))*f.albedo.w;
            ro=hit.position+direction*.03;
            continue;
        }
        if(f.emission.w<-.001) {
            direction=reflect(direction,hit.normal);
            energy*=max(f.albedo.rgb,vec3(0.0))*(-f.emission.w);
            ro=hit.position+direction*.03;
            continue;
        }
        // When the BVH hit is visible from the real camera, read the exact
        // rasterized/fully lit pixel (including GLB textures), not a second
        // flat-color shading model. Depth equality prevents unrelated
        // foreground objects being pasted into the reflection.
        vec4 projected=camera.vp*vec4(hit.position,1.0);
        if(projected.w>.0001) {
            vec3 ndc=projected.xyz/projected.w;
            vec2 screen=ndc.xy*.5+.5;
            if(all(greaterThanEqual(screen,vec2(.001))) &&
               all(lessThanEqual(screen,vec2(.999))) &&
               ndc.z>=0.0 && ndc.z<=1.0) {
                float expected_depth=.1+.9*ndc.z;
                float scene_depth=texture(opaque_depth,screen).r;
                if(abs(scene_depth-expected_depth)<.0003)
                    return energy*texture(lit_image,screen).rgb;
            }
        }
        // A ray outside the viewport still hits real world geometry and
        // receives direct and cached indirect illumination, not a probe.
        vec3 radiance=optics_surface_radiance(hit,direction);
        return energy*(radiance/(vec3(1.0)+max(radiance,vec3(0.0))));
    }
    return vec3(0.0);
}
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
