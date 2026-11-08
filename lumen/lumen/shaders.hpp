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
    ivec2 card=ivec2(int(face_id%uint(CARD_COLUMNS)),int(face_id/uint(CARD_COLUMNS)));
    vec2 card_uv=clamp(uv,vec2(.5/float(CARD_SIZE)),
                        vec2(1.0-.5/float(CARD_SIZE)));
    ivec2 texel=card*CARD_SIZE+ivec2(card_uv*float(CARD_SIZE));
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
    // Bilinear interpolation must remain within the face's own lit card.
    // Clamp to the texel centers so edge taps cannot bleed or go invalid.
    uv=clamp(uv,vec2(.5/float(LIT_CARD_SIZE)),
                vec2(1.0-.5/float(LIT_CARD_SIZE)));
    vec2 texel=uv*float(LIT_CARD_SIZE)-.5;
    ivec2 base=ivec2(floor(texel));
    vec2 fraction=fract(texel);
    ivec2 origin=ivec2(int(face_id%uint(CARD_COLUMNS)),int(face_id/uint(CARD_COLUMNS)))*LIT_CARD_SIZE;
    vec3 result=vec3(0.0);
    for(int y=0;y<2;y++)
    for(int x=0;x<2;x++)
    {
        ivec2 xy=clamp(base+ivec2(x,y),ivec2(0),ivec2(LIT_CARD_SIZE-1));
        float w=(x==0?1.0-fraction.x:fraction.x)*
                (y==0?1.0-fraction.y:fraction.y);
        result+=max(texelFetch(lit_cards,origin+xy,0).rgb,vec3(0.0))*w;
    }
    return result;
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

vec3 area_light_sample(Face lamp,vec3 position,vec3 normal,vec2 uv,
                       out bool valid,out bool visible)
{
    valid=false;
    visible=false;
    vec3 target=lamp.origin.xyz+lamp.u.xyz*uv.x+lamp.v.xyz*uv.y;
    vec3 delta=target-position;
    float distance2=dot(delta,delta);
    if(distance2<.0001)return vec3(0.0);
    float distance=sqrt(distance2);
    vec3 wi=delta/distance;
    float incoming=max(dot(normal,wi),0.0);
    float outgoing=max(dot(lamp.normal.xyz,-wi),0.0);
    if(incoming<=0.0 || outgoing<=0.0)return vec3(0.0);
    valid=true;

    // Bias just enough to avoid self-intersection without lifting contact
    // shadows away from the occluder. Aim at the light from the new origin.
    float bias=EPSILON*1.5+.0025*(1.0-incoming);
    vec3 origin=position+normal*bias;
    vec3 offset=target-origin;
    float travel=length(offset);
    if(travel<=EPSILON*4.0)return vec3(0.0);
    visible=!blocked(origin,offset/travel,travel-EPSILON*4.0);
    if(!visible)return vec3(0.0);
    return lamp.emission.rgb*(incoming*outgoing*lamp.metric.w/distance2);
}

vec3 direct_area_light(vec3 position,vec3 normal,vec3 albedo)
{
    Face lamp=faces[FACE_COUNT-1];
    vec3 radiance=vec3(0.0);
    int valid_count=0;
    int visible_count=0;

    // Four rays suffice in fully illuminated or fully shadowed regions.
    for(int y=0;y<2;y++)
    for(int x=0;x<2;x++)
    {
        bool valid,visible;
        vec2 uv=(vec2(x,y)+vec2(.5))*.5;
        radiance+=area_light_sample(lamp,position,normal,uv,valid,visible);
        if(valid)valid_count++;
        if(visible)visible_count++;
    }

    float sample_count=4.0;
    if(visible_count>0 && visible_count<valid_count)
    {
        // Re-evaluate the penumbra with sixteen fixed stratified emitter
        // samples. This is deterministic and adds no temporal noise.
        radiance=vec3(0.0);
        for(int y=0;y<4;y++)
        for(int x=0;x<4;x++)
        {
            bool valid,visible;
            vec2 uv=(vec2(x,y)+vec2(.5))*.25;
            radiance+=area_light_sample(lamp,position,normal,uv,valid,visible);
        }
        sample_count=16.0;
    }
    return albedo*(radiance/(sample_count*PI));
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
const int LIT_CARD_SIZE=128;
const int CARD_COLUMNS=4;
layout(set=0,binding=3) uniform sampler2D surface_cards;
layout(set=0,binding=11) uniform sampler2D direct_cards;
// Cubic B-spline weights give a smooth, nonnegative, energy-preserving
// footprint over a 4-texel span. Unlike ray retracing, this costs only
// 16 cache reads and cannot introduce sampling noise into rasterization.
vec4 shadow_weights(float t)
{
    float t2=t*t,t3=t2*t;
    float a=1.0-t;
    return vec4(a*a*a,
                4.0-6.0*t2+3.0*t3,
                1.0+3.0*t+3.0*t2-3.0*t3,
                t3)/6.0;
}

vec3 cached_direct(vec3 position,uint face_id)
{
    Face f=faces[face_id];
    vec3 d=position-f.origin.xyz;
    vec2 q=vec2(dot(d,f.u.xyz),dot(d,f.v.xyz));
    vec2 uv=clamp(vec2(q.x*f.metric.x+q.y*f.metric.y,
                       q.x*f.metric.y+q.y*f.metric.z),
                  vec2(.5/float(LIT_CARD_SIZE)),
                  vec2(1.0-.5/float(LIT_CARD_SIZE)));
    vec2 pixel=uv*float(LIT_CARD_SIZE)-.5;
    ivec2 center=ivec2(floor(pixel));
    vec2 fraction=fract(pixel);
    vec4 wx=shadow_weights(fraction.x);
    vec4 wy=shadow_weights(fraction.y);
    ivec2 origin=ivec2(int(face_id%uint(CARD_COLUMNS)),int(face_id/uint(CARD_COLUMNS)))*LIT_CARD_SIZE;
    vec3 value=vec3(0.0);
    for(int y=0;y<4;y++)
    for(int x=0;x<4;x++)
    {
        ivec2 local=clamp(center+ivec2(x-1,y-1),
                          ivec2(0),ivec2(LIT_CARD_SIZE-1));
        value+=max(texelFetch(direct_cards,origin+local,0).rgb,vec3(0.0))*
               wx[x]*wy[y];
    }
    return value;
}

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
    // All direct lighting comes from the Surface Cache. Reconstruct the
    // shadow edge with cubic filtering; no fragment-stage BVH traversal.
    // Screen Probe Gather continues to own all indirect lighting.
    vec3 direct=cached_direct(world_position,surface_index);
    hdr_color=vec4(max(direct,vec3(0.0)),float(surface_index+1u));
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
const int FACE_COUNT=16;
const int CARD_SIZE=64;
const int CARD_COLUMNS=4;

void main()
{
    ivec2 pixel=ivec2(gl_GlobalInvocationID.xy);
    if(any(greaterThanEqual(pixel,ivec2(CARD_SIZE*CARD_COLUMNS))))return;
    int face_index=(pixel.y/CARD_SIZE)*CARD_COLUMNS+(pixel.x/CARD_SIZE);
    if(face_index>=FACE_COUNT)
    {
        imageStore(cards,pixel,vec4(0.0));
        return;
    }
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
const int LIT_CARD_SIZE=128;
const int CARD_COLUMNS=4;
/* LUMEN_SHARED_TRACING */
/* LUMEN_SHARED_IRRADIANCE */
layout(rgba16f,set=0,binding=10) writeonly uniform image2D next_lighting;
layout(rgba16f,set=0,binding=11) writeonly uniform image2D direct_lighting;
layout(push_constant) uniform LightUpdate {
    uint card_to_update;
    uint enable_indirect;
    uint reserved0;
    uint reserved1;
} params;
void main()
{
    ivec2 local_pixel=ivec2(gl_GlobalInvocationID.xy);
    if(any(greaterThanEqual(local_pixel,ivec2(LIT_CARD_SIZE))))return;
    uint face_id=params.card_to_update;
    ivec2 pixel=ivec2(int(face_id%uint(CARD_COLUMNS)),int(face_id/uint(CARD_COLUMNS)))*LIT_CARD_SIZE+local_pixel;
    Face face=faces[face_id];
    vec2 uv=(vec2(local_pixel)+.5)/float(LIT_CARD_SIZE);
    vec3 position=face.origin.xyz+uv.x*face.u.xyz+uv.y*face.v.xyz;
    vec3 albedo=cached_albedo(position,face_id);
    vec3 normal=normalize(face.normal.xyz);
    vec3 direct=face.emission.rgb;
    vec3 radiance=direct;
    if(all(lessThanEqual(direct,vec3(0.0))))
    {
        direct=direct_area_light(position,normal,albedo);
        radiance=direct;
        if(params.enable_indirect!=0u)
            radiance+=albedo*indirect_irradiance(position,normal);
    }
    imageStore(direct_lighting,pixel,vec4(max(direct,vec3(0.0)),1.0));
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
const int LIT_CARD_SIZE=128;
const int CARD_COLUMNS=4;

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
#ifdef LUMEN_EXTERNAL_SCENE
    return false;
#else
    return inside_box(p,vec2(1.30,.65),vec2(.82,2.25),
                       vec2(2.40,2.72),vec2(2.90,1.14),1.65) ||
           inside_box(p,vec2(4.23,2.47),vec2(2.65,2.96),
                       vec2(3.14,4.56),vec2(4.72,4.06),3.30);
#endif
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

    // Interleave 32 low-discrepancy strata over successive visits to each
    // probe. Eight directions cover the sphere every visit, while 32
    // visits cover distinct directions instead of repeating random clusters.
    uint visit=(params.frame_number*updates+thread)/PROBE_COUNT;
    uint hash=(id+1u)*747796405u;
    random_bits(hash);
    float spin=2.0*PI*float(hash>>8)*(1.0/16777216.0);
    vec3 l0=vec3(0.0),lx=vec3(0.0),ly=vec3(0.0),lz=vec3(0.0);
    int samples=int(clamp(params.rays_per_probe,1u,16u));
    for(int ray=0;ray<samples;ray++)
    {
        uint sample_id=uint(ray)*32u+(visit%32u);
        float z=1.0-2.0*(float(sample_id)+.5)/float(samples*32);
        float a=spin+2.0*PI*fract(float(sample_id)*.61803398875);
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
            // Analytic emitter light is already in direct cards. World probes
            // gather only light reflected by non-emissive surfaces.
            if(dot(normal,-direction)>0.0 &&
               all(lessThanEqual(hit_face.emission.rgb,vec3(0.0))))
            {
                vec3 position=origin+direction*hit.distance;
                // Hit-lighting includes emissive energy and cached
                // reflected direct/indirect radiance from the surface card.
                incident=cached_lighting(position,face_id);
            }
        }
        incident=clamp(incident,vec3(0.0),vec3(32.0));
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
const int LIT_CARD_SIZE=128;
const int CARD_COLUMNS=4;
/* LUMEN_SHARED_TRACING */
/* LUMEN_SHARED_IRRADIANCE */
layout(set=0,binding=5) uniform sampler2D scene_color;
layout(set=0,binding=6) uniform sampler2D scene_depth;
// 8x8 octahedral directional atlas for each 8x8 screen probe.
layout(rgba16f,set=0,binding=7) uniform image2D screen_directions;
layout(rgba16f,set=0,binding=8) uniform image2D screen_meta;
layout(rgba16f,set=0,binding=10) uniform image2D screen_gather;
layout(rgba16f,set=0,binding=11) writeonly uniform image2D screen_specular;
layout(set=0,binding=12) uniform sampler2D direct_cards;
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
struct WorldSample {
    vec3 l0,lx,ly,lz;
};
WorldSample sample_world(vec3 position,vec3 normal)
{
    // Interpolate only valid, visible world probes on the receiving side
    // of the surface. Without this, probes behind the blocks/light leak
    // into the foreground and the ceiling.
    vec3 biased=position+normal*.075;
    vec3 coordinate=clamp(
        (biased-GRID_MIN)/(GRID_MAX-GRID_MIN)*vec3(PROBE_GRID-1),
        vec3(0.0),vec3(PROBE_GRID-1)-vec3(.001));
    ivec3 base=ivec3(floor(coordinate));
    vec3 fraction=fract(coordinate);
    WorldSample value;
    value.l0=vec3(0);value.lx=vec3(0);
    value.ly=vec3(0);value.lz=vec3(0);
    float sum=0.0;
    vec3 ray_origin=position+normal*.018;
    for(int z=0;z<2;z++)
    for(int y=0;y<2;y++)
    for(int x=0;x<2;x++)
    {
        ivec3 cell=base+ivec3(x,y,z);
        Probe p=probes[cell.x+12*cell.y+96*cell.z];
        if(p.l0.w<.5)continue;
        vec3 probe_pos=mix(GRID_MIN,GRID_MAX,
                            vec3(cell)/vec3(PROBE_GRID-1));
        vec3 to_probe=probe_pos-ray_origin;
        float length_to_probe=length(to_probe);
        if(length_to_probe<.02)continue;
        float facing=clamp((dot(normal,to_probe)/length_to_probe+.15)*2.0,
                           0.0,1.0);
        if(facing<=0.0)continue;
        float w=(x==0?1.0-fraction.x:fraction.x)*
                (y==0?1.0-fraction.y:fraction.y)*
                (z==0?1.0-fraction.z:fraction.z)*
                facing*facing;
        if(w<1.e-5)continue;
        if(blocked(ray_origin,to_probe/length_to_probe,
                   max(.001,length_to_probe-.02)))continue;
        value.l0+=p.l0.rgb*w;
        value.lx+=p.lx.rgb*w;
        value.ly+=p.ly.rgb*w;
        value.lz+=p.lz.rgb*w;
        sum+=w;
    }
    if(sum>1.e-5)
    {
        value.l0/=sum;value.lx/=sum;
        value.ly/=sum;value.lz/=sum;
    }
    return value;
}
vec3 distant_radiance(WorldSample world,vec3 direction)
{
    return max(world.l0+world.lx*direction.x+
               world.ly*direction.y+world.lz*direction.z,vec3(0.0));
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
        if(face_id<1u || face_id>uint(FACE_COUNT) || face_id==receiver)continue;
        if(dot(faces[face_id-1u].normal.xyz,-direction)<=.01)continue;
        vec2 ndc_center=(vec2(pixel)+.5)/vec2(size)*2.0-1.0;
        vec3 hit=camera.eye.xyz+
            (forward+camera.right.xyz*ndc_center.x*
                camera.projection.x*camera.projection.y-
                camera.up.xyz*ndc_center.y*camera.projection.x)*depth;
        // A projected depth sample is not a verified ray hit. Confirm it
        // matches the first BVH intersection in face and distance.
        vec3 separation=hit-origin;
        float ray_t=dot(separation,direction);
        float lateral=length(separation-direction*ray_t);
        if(ray_t<=EPSILON || ray_t>=limit ||
           lateral>max(.015,.007*ray_t))continue;
        RayHit actual=trace_scene(origin,direction,
                                  min(limit,ray_t+max(.08,.015*ray_t)),false);
        if(actual.overflow || actual.primitive==INVALID ||
           actual.primitive/2u!=face_id-1u ||
           abs(actual.distance-ray_t)>max(.05,.012*ray_t))
            continue;
        radiance=any(greaterThan(faces[face_id-1u].emission.rgb,vec3(0.0)))?
            vec3(0.0):
            cached_lighting(origin+direction*actual.distance,face_id-1u);
        return true;
    }
    return false;
}
vec3 scene_radiance(vec3 origin,vec3 direction,float max_distance)
{
    RayHit hit=trace_scene(origin,direction,max_distance,false);
    // This demo has no external illumination. Rays escaping the open front
    // must not receive unrelated local probe radiance.
    if(hit.primitive==INVALID || hit.overflow)
        return vec3(0.0);
    uint face_id=hit.primitive/2u;
    Face face=faces[face_id];
    if(dot(face.normal.xyz,-direction)<=0.0 ||
       any(greaterThan(face.emission.rgb,vec3(0.0))))
        return vec3(0.0);
    return cached_lighting(origin+direction*hit.distance,face_id);
}
vec3 directional_radiance(vec3 origin,vec3 direction,
                          float max_distance,uint face_id,
                          WorldSample world)
{
    vec3 result=vec3(0);
    if(screen_hit(origin,direction,max_distance,face_id,result))
        return result;
    return scene_radiance(origin,direction,max_distance);
}
// Integrate the ceiling emitter at screen-probe resolution, not at
// raster resolution. Indirect radiance intentionally excludes emitter
// hits, so direct rough specular needs a separate analytic term.
// Estimate source visibility using the direct-light Surface Cache.
vec3 area_specular(vec3 position,vec3 N,vec3 V,
                   vec3 albedo,float roughness,uint face_id)
{
    float ndv=max(dot(N,V),0.0);
    if(ndv<=.01)return vec3(0.0);
    Face surface=faces[face_id];
    vec3 d=position-surface.origin.xyz;
    vec2 q=vec2(dot(d,surface.u.xyz),dot(d,surface.v.xyz));
    vec2 uv=clamp(vec2(q.x*surface.metric.x+q.y*surface.metric.y,
                       q.x*surface.metric.y+q.y*surface.metric.z),
                  vec2(.5/float(LIT_CARD_SIZE)),
                  vec2(1.0-.5/float(LIT_CARD_SIZE)));
    vec2 atlas_uv=(vec2(float(face_id%uint(CARD_COLUMNS)),
                        float(face_id/uint(CARD_COLUMNS)))+uv)/float(CARD_COLUMNS);
    vec3 cached=max(textureLod(direct_cards,atlas_uv,0.0).rgb,vec3(0.0));
    Face lamp=faces[FACE_COUNT-1];
    float a=max(roughness*roughness,.04);
    float a2=a*a;
    float k=(roughness+1.0)*(roughness+1.0)*.125;
    float gv=ndv/max(ndv*(1.0-k)+k,.0001);
    vec3 unshadowed=vec3(0.0);
    vec3 highlight=vec3(0.0);
    for(int y=0;y<3;y++)
    for(int x=0;x<3;x++)
    {
        vec2 uvLight=(vec2(x,y)+.5)/3.0;
        vec3 delta=lamp.origin.xyz+
                   lamp.u.xyz*uvLight.x+lamp.v.xyz*uvLight.y-position;
        float d2=dot(delta,delta);
        if(d2<.0001)continue;
        vec3 L=delta*inversesqrt(d2);
        float ndl=max(dot(N,L),0.0);
        float emit=max(dot(lamp.normal.xyz,-L),0.0);
        if(ndl<=0.0 || emit<=0.0)continue;
        vec3 energy=lamp.emission.rgb*(ndl*emit*lamp.metric.w/(9.0*d2));
        unshadowed+=energy*albedo/PI;
        vec3 H=normalize(V+L);
        float ndh=max(dot(N,H),0.0);
        float vdh=max(dot(V,H),0.0);
        float term=ndh*ndh*(a2-1.0)+1.0;
        float D=a2/max(PI*term*term,.0001);
        float gl=ndl/max(ndl*(1.0-k)+k,.0001);
        float F=.04+.96*pow(1.0-vdh,5.0);
        float brdf=D*gv*gl*F/max(4.0*ndv*ndl,.0001);
        highlight+=energy*brdf;
    }
    const vec3 luma=vec3(.2126,.7152,.0722);
    float reference=dot(unshadowed,luma);
    if(reference<.0001)return vec3(0.0);
    float visibility=clamp(dot(cached,luma)/reference,0.0,1.0);
    return max(highlight*visibility,vec3(0.0));
}
void main()
{
    ivec2 tile=ivec2(gl_GlobalInvocationID.xy);
    ivec2 grid=imageSize(screen_gather);
    if(any(greaterThanEqual(tile,grid)))return;
    ivec2 size=textureSize(scene_color,0);
    ivec2 pixel=min(tile*8+ivec2(4),size-1);
    vec4 sample_color=texelFetch(scene_color,pixel,0);
    float z=texelFetch(scene_depth,pixel,0).r;
    uint id=uint(sample_color.a+.5);
    bool valid=id>=1u && id<=uint(FACE_COUNT) && z<.99999 &&
        all(lessThanEqual(faces[id-1u].emission.rgb,vec3(0)));
    if(!valid)
    {
        imageStore(screen_meta,tile,vec4(0));
        imageStore(screen_gather,tile,vec4(0));
        imageStore(screen_specular,tile,vec4(0));
        return;
    }
    float view_z=linear_depth(z);
    vec2 ndc=(vec2(pixel)+.5)/vec2(size)*2.0-1.0;
    vec3 forward=cross(camera.up.xyz,camera.right.xyz);
    vec3 position=camera.eye.xyz+
        (forward+camera.right.xyz*ndc.x*camera.projection.x*
         camera.projection.y-camera.up.xyz*ndc.y*camera.projection.x)*view_z;
    vec3 normal=normalize(faces[id-1u].normal.xyz);
    WorldSample world=sample_world(position,normal);
    bool reuse=false;
    vec3 history_anchor=position;
    if(camera.trace_parameters.z>.5)
    {
        vec4 old_meta=imageLoad(screen_meta,tile);
        vec4 old_gather=imageLoad(screen_gather,tile);
        reuse=abs(old_gather.a-float(id))<.1 &&
              distance(old_meta.xyz,position)<.10;
        // Keep the original anchor until history is rejected; otherwise
        // small per-frame motion can drift the history across the room.
        if(reuse)history_anchor=old_meta.xyz;
    }
    uint frame=uint(camera.trace_parameters.w);
    vec3 integrated=vec3(0);
    float total_weight=0.0;
    // Reuse the same directional radiance samples for broad specular.
    // Smooth mirror reflections require a separate tracing milestone.
    float roughness=clamp(faces[id-1u].albedo.w,.4,.95);
    bool glossy=roughness<.90;
    vec3 reflected=vec3(0.0);
    float fresnel=0.0;
    if(glossy)
    {
        vec3 V=normalize(camera.eye.xyz-position);
        float ndv=max(dot(normal,V),0.0);
        reflected=reflect(-V,normal);
        float one_minus=1.0-ndv;
        float p2=one_minus*one_minus;
        fresnel=.04+.96*p2*p2*one_minus;
    }
    vec3 specular_sum=vec3(0.0);
    float specular_weight=0.0;
    int rays=int(clamp(camera.trace_parameters.y,1.0,8.0));
    int start=(int(frame)*rays)%64;
    for(int i=0;i<64;i++)
    {
        ivec2 octant=ivec2(i%8,i/8);
        ivec2 atlas_pixel=tile*8+octant;
        vec2 uv=(vec2(octant)+.5)/8.0*2.0-1.0;
        vec3 direction=oct_decode(uv);
        float cosine=max(dot(normal,direction),0.0);
        if(cosine<=0.0)continue;
        vec4 old=reuse?imageLoad(screen_directions,atlas_pixel):vec4(0);
        float count=old.a;
        vec3 mean=old.rgb;
        if(((i-start+64)%64)<rays)
        {
            // Jitter *inside* each octahedral bin. Previously each frame
            // retraced exactly the same direction, so history never
            // reduced angular aliasing or high-energy emitter fireflies.
            uint h=uint(tile.x)*1664525u ^ uint(tile.y)*1013904223u ^
                   uint(i)*747796405u ^ frame*2891336453u;
            h^=h>>16; h*=2246822519u; h^=h>>13;
            float rx=float(h&65535u)/65536.0;
            float ry=float((h>>16)&65535u)/65536.0;
            vec2 jitter=(vec2(rx,ry)-.5)*(.23);
            vec3 sample_direction=oct_decode(clamp(uv+jitter,vec2(-.999),vec2(.999)));
            vec3 fresh=directional_radiance(position+normal*.012,
                         sample_direction,clamp(camera.trace_parameters.x,.5,16.0),id,world);
            // Reprojected directional samples are accumulated in probe space,
            // not in full-resolution screen pixels.
            count=min(count+1.0,24.0);
            mean+= (fresh-mean)/count;
        }
        imageStore(screen_directions,atlas_pixel,vec4(mean,count));

        // Treat low-sample bins as low-confidence observations. World
        // radiance acts as a prior until enough ray samples arrive.
        vec3 prior=distant_radiance(world,direction);
        float confidence=count/(count+8.0);
        vec3 Li=mix(prior,mean,confidence);
        float jacobian=pow(abs(direction.x)+abs(direction.y)+abs(direction.z),3.0);
        float weight=cosine*jacobian;
        integrated+=Li*weight;
        total_weight+=weight;
        if(glossy)
        {
            float r=max(dot(reflected,direction),0.0);
            float r2=r*r;
            float r4=r2*r2;
            float r8=r4*r4;
            float sharp=clamp((.90-roughness)/.45,0.0,1.0);
            float lobe=mix(r4,r8,sharp)*jacobian;
            specular_sum+=Li*lobe;
            specular_weight+=lobe;
        }
    }
    // Nonnegative irradiance is the sole indirect contribution. The world
    // cache supplies a prior until the traced directions have converged;
    // it is never subtracted from an independently evaluated raster term.
    vec3 incoming=total_weight>0.0?integrated/total_weight:vec3(0.0);
    vec3 indirect=max(incoming,vec3(0.0))*cached_albedo(position,id-1u);
    vec3 specular=specular_weight>1.e-5?
        max(specular_sum/specular_weight,vec3(0.0))*fresnel:vec3(0.0);
    if(glossy)
    {
        vec3 V=normalize(camera.eye.xyz-position);
        specular+=area_specular(position,normal,V,
                               cached_albedo(position,id-1u),
                               roughness,id-1u);
    }
    imageStore(screen_meta,tile,vec4(history_anchor,view_z));
    imageStore(screen_gather,tile,vec4(indirect,float(id)));
    imageStore(screen_specular,tile,vec4(specular,float(id)));
}
)glsl";


inline constexpr const char* screen_filter_comp=R"glsl(#version 450
layout(local_size_x=8,local_size_y=8,local_size_z=1) in;
layout(set=0,binding=0) uniform sampler2D raw_gather;
layout(set=0,binding=1) uniform sampler2D screen_meta;
layout(rgba16f,set=0,binding=2) writeonly uniform image2D filtered_gather;
layout(set=0,binding=3) uniform sampler2D raw_specular;
layout(rgba16f,set=0,binding=4) writeonly uniform image2D filtered_specular;
void main()
{
    ivec2 tile=ivec2(gl_GlobalInvocationID.xy);
    ivec2 size=imageSize(filtered_gather);
    if(any(greaterThanEqual(tile,size)))return;
    vec4 center=texelFetch(raw_gather,tile,0);
    vec4 geometry=texelFetch(screen_meta,tile,0);
    if(center.a<.5)
    {
        imageStore(filtered_gather,tile,vec4(0));
        imageStore(filtered_specular,tile,vec4(0));
        return;
    }

    // Filter *lighting probes* before full-resolution reconstruction,
    // respecting material identity and world-space hit distance.
    vec3 color=vec3(0.0);
    vec3 specular=vec3(0.0);
    float sum=0.0;
    float radius=max(.22,.085*geometry.w);
    for(int y=-1;y<=1;y++)
    for(int x=-1;x<=1;x++)
    {
        ivec2 neighbor=tile+ivec2(x,y);
        if(any(lessThan(neighbor,ivec2(0))) ||
           any(greaterThanEqual(neighbor,size)))continue;
        vec4 l=texelFetch(raw_gather,neighbor,0);
        if(abs(l.a-center.a)>.1)continue;
        vec4 g=texelFetch(screen_meta,neighbor,0);
        float dist=distance(g.xyz,geometry.xyz);
        if(dist>radius*2.0)continue;
        float spatial=exp(-float(x*x+y*y)*.5);
        float geometric=exp(-dist*dist/(2.0*radius*radius));
        float w=spatial*geometric;
        color+=l.rgb*w;
        specular+=texelFetch(raw_specular,neighbor,0).rgb*w;
        sum+=w;
    }
    imageStore(filtered_gather,tile,
               vec4(sum>0.0?color/sum:center.rgb,center.a));
    imageStore(filtered_specular,tile,
               vec4(sum>0.0?specular/sum:vec3(0.0),center.a));
}
)glsl";


inline constexpr const char* output_upsample_comp=R"glsl(#version 450
layout(local_size_x=8,local_size_y=8,local_size_z=1) in;
layout(set=0,binding=0) uniform sampler2D scene_color;
layout(set=0,binding=1) uniform sampler2D scene_depth;
layout(set=0,binding=2) uniform sampler2D probe_diffuse;
layout(set=0,binding=3) uniform sampler2D probe_reflections;
layout(set=0,binding=4) uniform sampler2D probe_meta;
layout(rgba16f,set=0,binding=5) writeonly uniform image2D diffuse_out;
layout(rgba16f,set=0,binding=6) writeonly uniform image2D reflections_out;

float linear_depth(float z)
{
    const float n=.05,f=200.0;
    return n*f/max(f-z*(f-n),.00001);
}
void main()
{
    ivec2 pixel=ivec2(gl_GlobalInvocationID.xy);
    ivec2 size=imageSize(diffuse_out);
    if(any(greaterThanEqual(pixel,size)))return;
    float id=texelFetch(scene_color,pixel,0).a;
    float z=texelFetch(scene_depth,pixel,0).r;
    if(id<=.5 || z>=.99999)
    {
        imageStore(diffuse_out,pixel,vec4(0.0));
        imageStore(reflections_out,pixel,vec4(0.0));
        return;
    }
    float depth=linear_depth(z);
    vec2 coord=(vec2(pixel)-vec2(4.0))/8.0;
    ivec2 origin=ivec2(floor(coord));
    vec2 fraction=fract(coord);
    ivec2 extent=textureSize(probe_diffuse,0);
    vec3 diffuse=vec3(0.0),specular=vec3(0.0);
    float sum=0.0;
    for(int y=0;y<2;y++)
    for(int x=0;x<2;x++)
    {
        ivec2 cell=origin+ivec2(x,y);
        if(any(lessThan(cell,ivec2(0))) ||
           any(greaterThanEqual(cell,extent)))continue;
        vec4 light=texelFetch(probe_diffuse,cell,0);
        if(abs(light.a-id)>.1)continue;
        vec4 meta=texelFetch(probe_meta,cell,0);
        float delta=abs(meta.a-depth);
        float tolerance=max(.10,.04*depth);
        if(delta>tolerance)continue;
        float weight=(x==0?1.0-fraction.x:fraction.x)*
                     (y==0?1.0-fraction.y:fraction.y);
        weight*=1.0-smoothstep(0.0,tolerance,delta);
        diffuse+=light.rgb*weight;
        specular+=texelFetch(probe_reflections,cell,0).rgb*weight;
        sum+=weight;
    }
    if(sum>.001)
    {
        diffuse=max(diffuse/sum,vec3(0.0));
        specular=max(specular/sum,vec3(0.0));
    }
    imageStore(diffuse_out,pixel,vec4(diffuse,id));
    imageStore(reflections_out,pixel,vec4(specular,id));
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
layout(set=0,binding=4) uniform sampler2D screen_specular;
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
        vec2 coord=(vec2(pixel)-vec2(4.0))/8.0;
        ivec2 origin=ivec2(floor(coord));
        vec2 fraction=fract(coord);
        ivec2 probe_size=textureSize(screen_gather,0);
        vec3 indirect=vec3(0.0);
        vec3 specular=vec3(0.0);
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
            indirect+=light.rgb*weight;
            specular+=texelFetch(screen_specular,tile,0).rgb*weight;
            sum+=weight;
        }
        if(sum>.001)
            hdr+=max((indirect+specular)/sum,vec3(0.0));
    }
    hdr=max(hdr,vec3(0.0));
    vec3 mapped=hdr/(vec3(1.0)+hdr);
    if(display.manual_gamma!=0u)
        mapped=pow(mapped,vec3(1.0/2.2));
    color_out=vec4(mapped,1.0);
}
)glsl";

} // namespace lumen::shaders
