#pragma once

// These GLSL sources belong to VulkanLumen, not the test application.
// Stage 2: original Cornell geometry, direct area-light shadows and composite.
namespace lumen::shaders {

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

void main()
{
    if(any(greaterThan(surface_emission,vec3(0.0))))
    {
        hdr_color=vec4(surface_emission,1.0);
        return;
    }
    vec3 normal=normalize(surface_normal);
    vec3 direct=direct_area_light(world_position,normal,
                                  cached_albedo(world_position,surface_index));
    hdr_color=vec4(max(direct,vec3(0.0)),1.0);
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
layout(rgba16f,set=0,binding=1) writeonly uniform image2D cards;

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
layout(push_constant) uniform Display {
    uint manual_gamma;
} display;
layout(location=0) in vec2 uv;
layout(location=0) out vec4 color_out;
void main()
{
    vec3 hdr=max(texture(color_image,uv).rgb,vec3(0.0));
    vec3 mapped=hdr/(vec3(1.0)+hdr);
    if(display.manual_gamma!=0u)
        mapped=pow(mapped,vec3(1.0/2.2));
    color_out=vec4(mapped,1.0);
}
)glsl";

} // namespace lumen::shaders
