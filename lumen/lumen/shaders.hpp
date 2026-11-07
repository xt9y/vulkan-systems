#pragma once

// These GLSL sources belong to VulkanLumen, not the test application.
// Stage 2: original Cornell geometry, direct area-light shadows and composite.
namespace lumen::shaders {

inline constexpr const char* geometry_vert=R"glsl(#version 450
layout(location=0) in vec3 position;
layout(location=1) in vec3 normal;
layout(location=2) in vec3 albedo;
layout(location=3) in vec3 emission;

layout(location=0) out vec3 world_position;
layout(location=1) out vec3 surface_normal;
layout(location=2) out vec3 surface_albedo;
layout(location=3) out vec3 surface_emission;

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
}
)glsl";

inline constexpr const char* geometry_frag=R"glsl(#version 450
layout(location=0) in vec3 world_position;
layout(location=1) in vec3 surface_normal;
layout(location=2) in vec3 surface_albedo;
layout(location=3) in vec3 surface_emission;
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
    // Fixed 2x2 stratified area-light integration:
    // deterministic soft shadows, no temporal denoiser or noise.
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
        if(blocked(position+normal*EPSILON*2.0,wi,
                   distance-EPSILON*4.0))
            continue;
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
                                  surface_albedo);
    hdr_color=vec4(max(direct,vec3(0.0)),1.0);
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
