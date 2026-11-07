#pragma once

// These GLSL sources belong to VulkanLumen, not the test application.
// Stage 2: original Cornell geometry, direct area-light shadows and composite.
namespace lumen::shaders {

inline constexpr const char* shared_tracing=R"glsl(
vec3 cached_albedo(vec3 position,uint face_id)
{
    Face f=faces[face_id];
    vec3 d=position-f.origin.xyz;
    vec2 q=vec2(dot(d,f.u.xyz),dot(d,f.v.xyz));
    vec2 uv=vec2(q.x*f.metric.x+q.y*f.metric.y,
                  q.x*f.metric.y+q.y*f.metric.z);
    ivec2 card=ivec2(int(face_id%4u),int(face_id/4u));
    ivec2 texel=card*CARD_SIZE+ivec2(clamp(uv,vec2(0.0),vec2(.9999))*float(CARD_SIZE));
    return texelFetch(surface_cards,texel,0).rgb;
}


#ifdef LUMEN_LIT_TEXTURE
layout(set=0,binding=9) uniform sampler2D lit_cards;
vec3 cached_lighting(vec3 position,uint face_id)
{
    Face face=faces[face_id];
    if(any(greaterThan(face.emission.rgb,vec3(0.0))))
        return face.emission.rgb;
    vec3 d=position-face.origin.xyz;
    vec2 q=vec2(dot(d,face.u.xyz),dot(d,face.v.xyz));
    vec2 uv=vec2(q.x*face.metric.x+q.y*face.metric.y,
                  q.x*face.metric.y+q.y*face.metric.z);
    ivec2 card=ivec2(int(face_id%4u),int(face_id/4u));
    ivec2 pixel=card*CARD_SIZE+
        ivec2(clamp(uv,vec2(0.0),vec2(.9999))*float(CARD_SIZE));
    return max(texelFetch(lit_cards,pixel,0).rgb,vec3(0.0));
}
#endif

// Uses the same std430 BVH node and triangle layout as VulkanBVH.
struct Node {
    vec4 lo,hi;
    uvec4 links; // left, right, parent, primitive (0xffffffff = interior)
};
struct Triangle {
    vec4 a,b,c;
};
layout(std430,set=0,binding=1) readonly buffer TraceNodes {
    Node nodes[];
};
layout(std430,set=0,binding=2) readonly buffer TraceTriangles {
    Triangle triangles[];
};
const uint INVALID=0xffffffffu;

bool box_hit(vec3 origin,vec3 direction,float limit,
             vec3 lower,vec3 upper)
{
    float near_t=EPSILON,far_t=limit;
    for(int axis=0;axis<3;axis++)
    {
        float d=direction[axis],o=origin[axis];
        if(abs(d)<0.0000001)
        {
            if(o<lower[axis] || o>upper[axis])return false;
        }
        else
        {
            float inverse=1.0/d;
            float a=(lower[axis]-o)*inverse;
            float b=(upper[axis]-o)*inverse;
            near_t=max(near_t,min(a,b));
            far_t=min(far_t,max(a,b));
            if(near_t>far_t)return false;
        }
    }
    return far_t>EPSILON;
}

struct RayHit {
    uint primitive;
    float distance;
    vec2 barycentric;
    bool overflow;
};

bool triangle_hit(vec3 origin,vec3 direction,float limit,Triangle tri,
                  out float distance,out vec2 barycentric)
{
    vec3 e1=tri.b.xyz-tri.a.xyz;
    vec3 e2=tri.c.xyz-tri.a.xyz;
    vec3 p=cross(direction,e2);
    float det=dot(e1,p);
    if(abs(det)<0.00000001)return false;
    float inv=1.0/det;
    vec3 t=origin-tri.a.xyz;
    float u=dot(t,p)*inv;
    if(u<0.0 || u>1.0)return false;
    vec3 q=cross(t,e1);
    float v=dot(direction,q)*inv;
    if(v<0.0 || u+v>1.0)return false;
    float hit_t=dot(e2,q)*inv;
    if(hit_t<=EPSILON || hit_t>=limit)return false;
    distance=hit_t;
    barycentric=vec2(u,v);
    return true;
}

RayHit trace_scene(vec3 origin,vec3 direction,float limit,bool any_hit)
{
    // Closest- or any-hit traversal of the same arbitrary triangle BVH.
    RayHit result;
    result.primitive=INVALID;
    result.distance=limit;
    result.barycentric=vec2(0.0);
    result.overflow=false;
    uint stack[64];
    int count=1;
    stack[0]=0u;
    while(count>0)
    {
        uint index=stack[--count];
        Node n=nodes[index];
        if(!box_hit(origin,direction,result.distance,n.lo.xyz,n.hi.xyz))
            continue;
        if(n.links.w!=INVALID)
        {
            float distance;
            vec2 barycentric;
            if(triangle_hit(origin,direction,result.distance,
                            triangles[n.links.w],distance,barycentric))
            {
                result.primitive=n.links.w;
                result.distance=distance;
                result.barycentric=barycentric;
                if(any_hit)return result;
            }
        }
        else
        {
            if(count+2>64){result.overflow=true;break;}
            stack[count++]=n.links.x;
            stack[count++]=n.links.y;
        }
    }
    return result;
}

bool blocked(vec3 origin,vec3 direction,float limit)
{
    RayHit hit=trace_scene(origin,direction,limit,true);
    return hit.primitive!=INVALID || hit.overflow;
}

vec3 direct_area_light(vec3 position,vec3 normal,vec3 albedo)
{
    Face lamp=faces[FACE_COUNT-1];
    vec3 radiance=vec3(0.0);
    // M1 fast path: one BVH visibility query for this surface. Four geometry
    // samples still integrate the emitter's smooth angular falloff.
    // The shadow silhouette is intentionally harder than four distinct rays.
    vec3 center=lamp.origin.xyz+.5*lamp.u.xyz+.5*lamp.v.xyz;
    vec3 to_center=center-position;
    float center_distance=length(to_center);
    if(center_distance<EPSILON)return vec3(0.0);
    vec3 center_direction=to_center/center_distance;
    if(dot(normal,center_direction)<=0.0 ||
       dot(lamp.normal.xyz,-center_direction)<=0.0)
        return vec3(0.0);
    if(blocked(position+normal*EPSILON*2.0,
               center_direction,center_distance-EPSILON*4.0))
        return vec3(0.0);
    // Fixed 2x2 emitter quadrature; no stochastic noise or denoising.
    for(int y=0;y<2;y++)
    for(int x=0;x<2;x++)
    {
        vec2 uv=(vec2(x,y)+vec2(.5))*.5;
        vec3 target=lamp.origin.xyz+lamp.u.xyz*uv.x+
                    lamp.v.xyz*uv.y;
        vec3 delta=target-position;
        float distance2=dot(delta,delta);
        if(distance2<0.0001)continue;
        float distance=sqrt(distance2);
        vec3 wi=delta/distance;
        float incoming=max(dot(normal,wi),0.0);
        float outgoing=max(dot(lamp.normal.xyz,-wi),0.0);
        if(incoming<=0.0 || outgoing<=0.0)continue;
        radiance+=lamp.emission.rgb*
                  (incoming*outgoing*lamp.metric.w/distance2);
    }
    return albedo*(radiance/(4.0*PI));
}

)glsl";

inline constexpr const char* shared_irradiance=R"glsl(
struct Probe {
    vec4 l0,lx,ly,lz;
};
layout(std430,set=0,binding=4) readonly buffer RadianceProbes {
    Probe probes[];
};
const ivec3 PROBE_GRID=ivec3(12,8,12);
const vec3 GRID_MIN=vec3(.25,.24,.25);
const vec3 GRID_MAX=vec3(5.30,5.22,5.34);

vec3 indirect_irradiance(vec3 position,vec3 normal)
{
    vec3 biased=position+normal*.075;
    vec3 coordinate=clamp(
        (biased-GRID_MIN)/(GRID_MAX-GRID_MIN)*vec3(PROBE_GRID-1),
        vec3(0.0),vec3(PROBE_GRID-1)-vec3(.001));
    ivec3 base=ivec3(floor(coordinate));
    vec3 fraction=fract(coordinate);
    vec3 sum=vec3(0.0);
    float weights=0.0;
    for(int z=0;z<2;z++)
    for(int y=0;y<2;y++)
    for(int x=0;x<2;x++)
    {
        ivec3 cell=base+ivec3(x,y,z);
        uint id=uint(cell.x+cell.y*12+cell.z*96);
        Probe probe=probes[id];
        if(probe.l0.w<.5)continue;
        vec3 probe_pos=mix(GRID_MIN,GRID_MAX,
                           vec3(cell)/vec3(PROBE_GRID-1));
        vec3 toward=probe_pos-position;
        float distance=length(toward);
        // Normal-biased interpolation avoids sampling deeply behind walls
        // and rejects cache probes underneath the receiver surface.
        float facing=clamp((dot(normal,toward)/max(distance,.01)+.15)*2.0,
                            0.0,1.0);
        float weight=(x==0?1.0-fraction.x:fraction.x)*
                     (y==0?1.0-fraction.y:fraction.y)*
                     (z==0?1.0-fraction.z:fraction.z);
        weight*=max(.002,facing*facing);
        vec3 irradiance=probe.l0.rgb+
            (2.0/3.0)*(probe.lx.rgb*normal.x+
                        probe.ly.rgb*normal.y+
                        probe.lz.rgb*normal.z);
        sum+=max(irradiance,vec3(0.0))*weight;
        weights+=weight;
    }
    return weights>.0001?sum/weights:vec3(0.0);
}

)glsl";

inline constexpr const char* geometry_vert=R"glsl(#version 450
layout(location=0) in vec3 position;
layout(location=1) in vec3 normal;
layout(location=2) in vec3 albedo;
layout(location=3) in vec3 emission;
layout(location=4) in uint face_index;

layout(location=0) out vec3 world_position;
layout(location=1) out vec3 surface_normal;
layout(location=2) out vec3 surface_albedo;
layout(location=3) out vec3 surface_emission;
layout(location=4) flat out uint surface_index;

layout(push_constant) uniform Camera {
    vec4 eye;
    vec4 right;
    vec4 up;
    vec4 projection; // tan(fov/2), aspect, near, far
} camera;

void main()
{
    vec3 relative=position-camera.eye.xyz;
    vec3 forward=cross(camera.up.xyz,camera.right.xyz);
    float view_x=dot(relative,camera.right.xyz);
    float view_y=dot(relative,camera.up.xyz);
    float view_z=dot(relative,forward);
    float n=camera.projection.z;
    float f=camera.projection.w;
    float z=view_z*f/(f-n)-n*f/(f-n);
    gl_Position=vec4(view_x/(camera.projection.x*camera.projection.y),
                    -view_y/camera.projection.x,z,view_z);
    world_position=position;
    surface_normal=normal;
    surface_albedo=albedo;
    surface_emission=emission;
    surface_index=face_index;
}
)glsl";

inline constexpr const char* geometry_frag=R"glsl(#version 450
layout(location=0) in vec3 world_position;
layout(location=1) in vec3 surface_normal;
layout(location=2) in vec3 surface_albedo;
layout(location=3) in vec3 surface_emission;
layout(location=4) flat in uint surface_index;
layout(location=0) out vec4 hdr_color;

// These are the same sixteen original Cornell faces as path-tracer-bak.
// Precomputed inverse Gram coefficients make shadow intersections cheap.
struct Face {
    vec4 origin,u,v,normal,albedo,emission,metric;
};
layout(std430,set=0,binding=0) readonly buffer CornellFaces {
    Face faces[];
};
const float PI=3.14159265358979323846;
const float EPSILON=0.0015;
const int FACE_COUNT=16;
const int CARD_SIZE=64;
layout(set=0,binding=3) uniform sampler2D surface_cards;

/* LUMEN_SHARED_TRACING */

/* LUMEN_SHARED_IRRADIANCE */

void main()
{
    if(any(greaterThan(surface_emission,vec3(0.0))))
    {
        hdr_color=vec4(surface_emission,float(surface_index+1u));
        return;
    }
    vec3 normal=normalize(surface_normal);
    vec3 albedo=cached_albedo(world_position,surface_index);
    vec3 direct=direct_area_light(world_position,normal,albedo);
    vec3 indirect=albedo*indirect_irradiance(world_position,normal);
    hdr_color=vec4(max(direct+indirect,vec3(0.0)),float(surface_index+1u));
}
)glsl";


inline constexpr const char* cards_comp=R"glsl(#version 450
layout(local_size_x=8,local_size_y=8,local_size_z=1) in;
struct Face {
    vec4 origin,u,v,normal,albedo,emission,metric;
};
layout(std430,set=0,binding=0) readonly buffer CornellFaces {
    Face faces[];
};
layout(rgba8,set=0,binding=1) writeonly uniform image2D cards;

void main()
{
    ivec2 pixel=ivec2(gl_GlobalInvocationID.xy);
    if(any(greaterThanEqual(pixel,ivec2(256))))return;
    int face_index=(pixel.y/64)*4+(pixel.x/64);
    Face f=faces[face_index];
    // Stage 4 material cards: colors/emission stay in world-independent UV
    // space. Future stages will add lighting caches and card recapture.
    imageStore(cards,pixel,vec4(f.albedo.rgb,1.0));
}
)glsl";


inline constexpr const char* lighting_comp=R"glsl(#version 450
layout(local_size_x=8,local_size_y=8,local_size_z=1) in;
struct Face { vec4 origin,u,v,normal,albedo,emission,metric; };
layout(std430,set=0,binding=0) readonly buffer CornellFaces { Face faces[]; };
layout(set=0,binding=3) uniform sampler2D surface_cards;
const float PI=3.14159265358979323846;
const float EPSILON=.0015;
const int FACE_COUNT=16;
const int CARD_SIZE=64;
/* LUMEN_SHARED_TRACING */
/* LUMEN_SHARED_IRRADIANCE */
layout(rgba16f,set=0,binding=10) writeonly uniform image2D next_lighting;
layout(push_constant) uniform LightUpdate {
    uint card_to_update;
    uint enable_indirect;
    uint reserved0;
    uint reserved1;
} params;
void main()
{
    ivec2 local_pixel=ivec2(gl_GlobalInvocationID.xy);
    if(any(greaterThanEqual(local_pixel,ivec2(64))))return;
    uint face_id=params.card_to_update;
    ivec2 pixel=ivec2(int(face_id%4u),int(face_id/4u))*64+local_pixel;
    Face face=faces[face_id];
    vec2 uv=(vec2(local_pixel)+.5)/64.0;
    vec3 position=face.origin.xyz+uv.x*face.u.xyz+uv.y*face.v.xyz;
    vec3 albedo=cached_albedo(position,face_id);
    vec3 normal=normalize(face.normal.xyz);
    vec3 radiance=face.emission.rgb;
    if(all(lessThanEqual(radiance,vec3(0.0))))
    {
        radiance=direct_area_light(position,normal,albedo);
        if(params.enable_indirect!=0u)
            radiance+=albedo*indirect_irradiance(position,normal);
    }
    imageStore(next_lighting,pixel,vec4(max(radiance,vec3(0.0)),1.0));
}
)glsl";

inline constexpr const char* radiance_comp=R"glsl(#version 450
layout(local_size_x=64,local_size_y=1,local_size_z=1) in;
struct Face {
    vec4 origin,u,v,normal,albedo,emission,metric;
};
layout(std430,set=0,binding=0) readonly buffer CornellFaces {
    Face faces[];
};
layout(set=0,binding=3) uniform sampler2D surface_cards;
const float PI=3.14159265358979323846;
const float EPSILON=0.0015;
const int FACE_COUNT=16;
const int CARD_SIZE=64;

#define LUMEN_LIT_TEXTURE
/* LUMEN_SHARED_TRACING */

struct Probe {
    vec4 l0,lx,ly,lz;
};
layout(std430,set=0,binding=4) buffer RadianceProbes {
    Probe probes[];
};
layout(push_constant) uniform Update {
    uint frame_number;
    uint revision;
    uint updates_per_frame;
    uint rays_per_probe;
} params;

const ivec3 PROBE_GRID=ivec3(12,8,12);
const uint PROBE_COUNT=1152u;
const vec3 GRID_MIN=vec3(.25,.24,.25);
const vec3 GRID_MAX=vec3(5.30,5.22,5.34);

uint random_bits(inout uint state)
{
    state ^= state<<13;
    state ^= state>>17;
    state ^= state<<5;
    return state;
}
float random01(inout uint state)
{
    return float(random_bits(state)>>8)*(1.0/16777216.0);
}
vec3 probe_position(ivec3 cell)
{
    return mix(GRID_MIN,GRID_MAX,
               vec3(cell)/vec3(PROBE_GRID-1));
}
float cross2(vec2 a,vec2 b)
{
    return a.x*b.y-a.y*b.x;
}
bool inside_box(vec3 p,vec2 a,vec2 b,vec2 c,vec2 d,float top)
{
    if(p.y<0.0 || p.y>top)return false;
    vec2 v=p.xz;
    float s0=cross2(b-a,v-a);
    float s1=cross2(c-b,v-b);
    float s2=cross2(d-c,v-c);
    float s3=cross2(a-d,v-d);
    return (s0<=0.0 && s1<=0.0 && s2<=0.0 && s3<=0.0) ||
           (s0>=0.0 && s1>=0.0 && s2>=0.0 && s3>=0.0);
}
bool occupied(vec3 p)
{
    return inside_box(p,vec2(1.30,.65),vec2(.82,2.25),
                      vec2(2.40,2.72),vec2(2.90,1.14),1.65) ||
           inside_box(p,vec2(4.23,2.47),vec2(2.65,2.96),
                      vec2(3.14,4.56),vec2(4.72,4.06),3.30);
}
void main()
{
    uint thread=gl_GlobalInvocationID.x;
    uint updates=clamp(params.updates_per_frame,1u,PROBE_COUNT);
    if(thread>=updates)return;
    uint id=(params.frame_number*updates+thread)%PROBE_COUNT;
    ivec3 cell=ivec3(int(id%12u),int((id/12u)%8u),int(id/96u));
    vec3 origin=probe_position(cell);
    if(occupied(origin))
    {
        probes[id].l0=vec4(0.0);
        probes[id].lx=vec4(0.0);
        probes[id].ly=vec4(0.0);
        probes[id].lz=vec4(0.0);
        return;
    }

    uint seed=(id+1u)*747796405u ^ (params.frame_number+1u)*2891336453u;
    random_bits(seed);
    vec3 l0=vec3(0.0),lx=vec3(0.0),ly=vec3(0.0),lz=vec3(0.0);
    float spin=2.0*PI*random01(seed);
    int samples=int(clamp(params.rays_per_probe,1u,16u));
    for(int ray=0;ray<samples;ray++)
    {
        // Uniform-sphere stratification with per-update rotation.
        float z=1.0-2.0*(float(ray)+random01(seed))/float(samples);
        float a=spin+2.3999632*float(ray)+6.2831853*random01(seed);
        float radial=sqrt(max(0.0,1.0-z*z));
        vec3 direction=vec3(cos(a)*radial,sin(a)*radial,z);
        RayHit hit=trace_scene(origin,direction,20.0,false);
        vec3 incident=vec3(0.0);
        if(hit.primitive!=INVALID)
        {
            uint face_id=hit.primitive/2u;
            Face hit_face=faces[face_id];
            vec3 normal=hit_face.normal.xyz;
            // Discard backfaces. The room surfaces face inward; box faces
            // point outward, so backface hits indicate unsuitable probes.
            if(dot(normal,-direction)>0.0)
            {
                vec3 position=origin+direction*hit.distance;
                // Hit-lighting includes emissive energy and cached
                // reflected direct/indirect radiance from the surface card.
                incident=cached_lighting(position,face_id);
            }
        }
        incident=clamp(incident,vec3(0.0),vec3(8.0));
        l0+=incident;
        lx+=incident*direction.x;
        ly+=incident*direction.y;
        lz+=incident*direction.z;
    }

    float inv=1.0/float(samples);
    l0*=inv;
    lx*=3.0*inv;
    ly*=3.0*inv;
    lz*=3.0*inv;
    Probe previous=probes[id];
    float count=min(previous.l0.w+1.0,32.0);
    float blend=1.0/count;
    probes[id].l0=vec4(mix(previous.l0.rgb,l0,blend),count);
    probes[id].lx=vec4(mix(previous.lx.rgb,lx,blend),0.0);
    probes[id].ly=vec4(mix(previous.ly.rgb,ly,blend),0.0);
    probes[id].lz=vec4(mix(previous.lz.rgb,lz,blend),0.0);
}
)glsl";


inline constexpr const char* screen_comp=R"glsl(#version 450
layout(local_size_x=8,local_size_y=8,local_size_z=1) in;
struct Face {vec4 origin,u,v,normal,albedo,emission,metric;};
layout(std430,set=0,binding=0) readonly buffer CornellFaces { Face faces[]; };
layout(set=0,binding=3) uniform sampler2D surface_cards;
#define LUMEN_LIT_TEXTURE
const float PI=3.14159265358979323846;
const float EPSILON=.0015;
const int FACE_COUNT=16;
const int CARD_SIZE=64;
/* LUMEN_SHARED_TRACING */
/* LUMEN_SHARED_IRRADIANCE */
layout(set=0,binding=5) uniform sampler2D scene_color;
layout(set=0,binding=6) uniform sampler2D scene_depth;
// 8x8 octahedral directional atlas for each 16x16 screen probe.
layout(rgba16f,set=0,binding=7) uniform image2D screen_directions;
layout(rgba16f,set=0,binding=8) uniform image2D screen_meta;
layout(rgba16f,set=0,binding=10) uniform image2D screen_gather;
layout(push_constant) uniform Camera {
    vec4 eye,right,up,projection;
    vec4 trace_parameters; // range, reserved, history valid, frame index
} camera;
float linear_depth(float z)
{
    float n=camera.projection.z,f=camera.projection.w;
    return n*f/max(f-z*(f-n),.00001);
}
vec3 oct_decode(vec2 e)
{
    vec3 v=vec3(e,1.0-abs(e.x)-abs(e.y));
    if(v.z<0.0)
        v.xy=(1.0-abs(v.yx))*sign(v.xy);
    return normalize(v);
}
vec3 distant_radiance(vec3 position,vec3 direction)
{
    vec3 coordinate=clamp(
        (position-GRID_MIN)/(GRID_MAX-GRID_MIN)*vec3(PROBE_GRID-1),
        vec3(0),vec3(PROBE_GRID-1));
    ivec3 cell=ivec3(round(coordinate));
    Probe p=probes[cell.x+12*cell.y+96*cell.z];
    if(p.l0.w<.5)return vec3(0);
    return max(p.l0.rgb+p.lx.rgb*direction.x+
               p.ly.rgb*direction.y+p.lz.rgb*direction.z,vec3(0));
}
bool screen_hit(vec3 origin,vec3 direction,float limit,
                uint receiver,out vec3 radiance)
{
    vec3 forward=cross(camera.up.xyz,camera.right.xyz);
    ivec2 size=textureSize(scene_depth,0);
    for(int i=0;i<8;i++)
    {
        float q=float(i+1)/8.0;
        vec3 relative=origin+direction*(limit*q*q)-camera.eye.xyz;
        float vz=dot(relative,forward);
        if(vz<camera.projection.z || vz>camera.projection.w)return false;
        vec2 ndc=vec2(
            dot(relative,camera.right.xyz)/
                (vz*camera.projection.x*camera.projection.y),
           -dot(relative,camera.up.xyz)/(vz*camera.projection.x));
        if(any(greaterThan(abs(ndc),vec2(.98))))return false;
        ivec2 pixel=clamp(ivec2((ndc*.5+.5)*vec2(size)),
                          ivec2(0),size-1);
        float z=texelFetch(scene_depth,pixel,0).r;
        if(z>=.99999)continue;
        float depth=linear_depth(z);
        if(vz<depth || vz-depth>max(.08,.025*limit))continue;
        uint face_id=uint(texelFetch(scene_color,pixel,0).a+.5);
        if(face_id<1u || face_id>16u || face_id==receiver)continue;
        if(dot(faces[face_id-1u].normal.xyz,-direction)<=.01)continue;
        vec2 ndc_center=(vec2(pixel)+.5)/vec2(size)*2.0-1.0;
        vec3 hit=camera.eye.xyz+
            (forward+camera.right.xyz*ndc_center.x*
                camera.projection.x*camera.projection.y-
                camera.up.xyz*ndc_center.y*camera.projection.x)*depth;
        radiance=cached_lighting(hit,face_id-1u);
        return true;
    }
    return false;
}
vec3 scene_radiance(vec3 origin,vec3 direction,float max_distance)
{
    RayHit hit=trace_scene(origin,direction,max_distance,false);
    if(hit.primitive==INVALID || hit.overflow)
        return distant_radiance(origin,direction);
    uint face_id=hit.primitive/2u;
    Face face=faces[face_id];
    if(dot(face.normal.xyz,-direction)<=0.0)return vec3(0);
    return cached_lighting(origin+direction*hit.distance,face_id);
}
vec3 directional_radiance(vec3 origin,vec3 direction,
                          float max_distance,uint face_id)
{
    vec3 result=vec3(0);
    if(screen_hit(origin,direction,max_distance,face_id,result))
        return result;
    return scene_radiance(origin,direction,max_distance);
}
void main()
{
    ivec2 tile=ivec2(gl_GlobalInvocationID.xy);
    ivec2 grid=imageSize(screen_gather);
    if(any(greaterThanEqual(tile,grid)))return;
    ivec2 size=textureSize(scene_color,0);
    ivec2 pixel=min(tile*16+ivec2(8),size-1);
    vec4 sample_color=texelFetch(scene_color,pixel,0);
    float z=texelFetch(scene_depth,pixel,0).r;
    uint id=uint(sample_color.a+.5);
    bool valid=id>=1u && id<=16u && z<.99999 &&
        all(lessThanEqual(faces[id-1u].emission.rgb,vec3(0)));
    if(!valid)
    {
        imageStore(screen_meta,tile,vec4(0));
        imageStore(screen_gather,tile,vec4(0));
        return;
    }
    float view_z=linear_depth(z);
    vec2 ndc=(vec2(pixel)+.5)/vec2(size)*2.0-1.0;
    vec3 forward=cross(camera.up.xyz,camera.right.xyz);
    vec3 position=camera.eye.xyz+
        (forward+camera.right.xyz*ndc.x*camera.projection.x*
         camera.projection.y-camera.up.xyz*ndc.y*camera.projection.x)*view_z;
    vec3 normal=normalize(faces[id-1u].normal.xyz);
    bool reuse=false;
    if(camera.trace_parameters.z>.5)
    {
        vec4 old_meta=imageLoad(screen_meta,tile);
        vec4 old_gather=imageLoad(screen_gather,tile);
        reuse=abs(old_gather.a-float(id))<.1 &&
              distance(old_meta.xyz,position)<.10;
    }
    uint frame=uint(camera.trace_parameters.w);
    int phase=int(frame%16u);
    vec3 integrated=vec3(0);
    float total_weight=0.0;
    for(int i=0;i<64;i++)
    {
        ivec2 octant=ivec2(i%8,i/8);
        ivec2 atlas_pixel=tile*8+octant;
        vec2 uv=(vec2(octant)+.5)/8.0*2.0-1.0;
        vec3 direction=oct_decode(uv);
        float cosine=max(dot(normal,direction),0.0);
        if(cosine<=0.0)continue;
        vec4 old=vec4(0);
        if(reuse)old=imageLoad(screen_directions,atlas_pixel);
        vec3 Li=old.a>.5?old.rgb:distant_radiance(position,direction);
        float count=old.a;
        if((i%16)==phase)
        {
            vec3 fresh=directional_radiance(position+normal*.012,
                         direction,clamp(camera.trace_parameters.x,.5,8.0),id);
            count=min(count+1.0,24.0);
            Li=mix(Li,fresh,1.0/count);
        }
        imageStore(screen_directions,atlas_pixel,vec4(Li,count));
        // Octahedral solid-angle Jacobian and cosine Lambertian weighting.
        float area=pow(abs(direction.x)+abs(direction.y)+abs(direction.z),3.0);
        float weight=cosine*area;
        integrated+=Li*weight;
        total_weight+=weight;
    }
    vec3 incoming=total_weight>0.0?integrated/total_weight:vec3(0);
    vec3 base=indirect_irradiance(position,normal);
    vec3 albedo=cached_albedo(position,id-1u);
    // Both terms are hemispherically integrated incoming radiance (E/pi),
    // unlike the former arbitrary one-ray radiance minus irradiance.
    vec3 delta=(incoming-base)*albedo;
    imageStore(screen_meta,tile,vec4(position,view_z));
    imageStore(screen_gather,tile,vec4(delta,float(id)));
}
)glsl";

inline constexpr const char* composite_vert=R"glsl(#version 450
layout(location=0) out vec2 uv;
void main()
{
    const vec2 triangle[3]=vec2[3](
        vec2(-1.0,-1.0),vec2(3.0,-1.0),vec2(-1.0,3.0));
    vec2 p=triangle[gl_VertexIndex];
    uv=p*0.5+0.5;
    gl_Position=vec4(p,0.0,1.0);
}
)glsl";

inline constexpr const char* composite_frag=R"glsl(#version 450
layout(set=0,binding=0) uniform sampler2D color_image;
layout(set=0,binding=1) uniform sampler2D depth_image;
layout(set=0,binding=2) uniform sampler2D screen_gather;
layout(set=0,binding=3) uniform sampler2D screen_meta;
layout(push_constant) uniform Display {
    uint manual_gamma;
} display;
layout(location=0) in vec2 uv;
layout(location=0) out vec4 color_out;

float linear_depth(float z)
{
    const float n=.05,f=200.0;
    return n*f/max(f-z*(f-n),.00001);
}
void main()
{
    vec3 hdr=texture(color_image,uv).rgb;
    ivec2 size=textureSize(color_image,0);
    ivec2 pixel=clamp(ivec2(uv*vec2(size)),ivec2(0),size-1);
    float id=texelFetch(color_image,pixel,0).a;
    float z=texelFetch(depth_image,pixel,0).r;
    if(id>.5 && z<.99999)
    {
        float distance=linear_depth(z);
        vec2 coord=(vec2(pixel)-vec2(8.0))/16.0;
        ivec2 origin=ivec2(floor(coord));
        vec2 fraction=fract(coord);
        ivec2 probe_size=textureSize(screen_gather,0);
        vec3 correction=vec3(0.0);
        float sum=0.0;
        for(int y=0;y<2;y++)
        for(int x=0;x<2;x++)
        {
            ivec2 tile=origin+ivec2(x,y);
            if(any(lessThan(tile,ivec2(0)))||
               any(greaterThanEqual(tile,probe_size)))continue;
            vec4 light=texelFetch(screen_gather,tile,0);
            if(abs(light.a-id)>.1)continue;
            vec4 meta=texelFetch(screen_meta,tile,0);
            float difference=abs(meta.a-distance);
            float tolerance=max(.10,.04*distance);
            if(difference>tolerance)continue;
            float weight=(x==0?1.0-fraction.x:fraction.x)*
                         (y==0?1.0-fraction.y:fraction.y);
            weight*=1.0-smoothstep(0.0,tolerance,difference);
            correction+=light.rgb*weight;
            sum+=weight;
        }
        if(sum>.001)
            hdr+=correction/sum;
    }
    hdr=max(hdr,vec3(0.0));
    vec3 mapped=hdr/(vec3(1.0)+hdr);
    if(display.manual_gamma!=0u)
        mapped=pow(mapped,vec3(1.0/2.2));
    color_out=vec4(mapped,1.0);
}
)glsl";

} // namespace lumen::shaders
