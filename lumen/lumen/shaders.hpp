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

bool intersects_bounds(vec3 ro,vec3 rd,float limit,
                       vec3 lower,vec3 upper)
{
    float lo=0.0,hi=limit;
    for(int axis=0;axis<3;axis++)
    {
        if(abs(rd[axis])<0.000001)
        {
            if(ro[axis]<lower[axis] || ro[axis]>upper[axis])
                return false;
        }
        else
        {
            float a=(lower[axis]-ro[axis])/rd[axis];
            float b=(upper[axis]-ro[axis])/rd[axis];
            lo=max(lo,min(a,b));
            hi=min(hi,max(a,b));
            if(hi<lo)return false;
        }
    }
    return hi>EPSILON;
}

bool blocked(vec3 ro,vec3 rd,float limit)
{
    // The enclosing Cornell room is convex: only the two internal blocks
    // can occlude an interior point's segment to the ceiling emitter.
    // Their conservative AABBs reject most rays before face intersections.
    bool small=intersects_bounds(ro,rd,limit,
                  vec3(.80,-.01,.63),vec3(2.92,1.67,2.74));
    bool tall=intersects_bounds(ro,rd,limit,
                  vec3(2.63,-.01,2.45),vec3(4.74,3.32,4.58));
    if(!small && !tall)return false;
    for(int i=5;i<15;i++)
    {
        if((i<10 && !small) || (i>=10 && !tall))continue;
        Face f=faces[i];
        float denom=dot(f.normal.xyz,rd);
        if(abs(denom)<0.000001)continue;
        float t=dot(f.origin.xyz-ro,f.normal.xyz)/denom;
        if(t<=EPSILON || t>=limit)continue;
        vec3 d=ro+rd*t-f.origin.xyz;
        vec2 q=vec2(dot(d,f.u.xyz),dot(d,f.v.xyz));
        float x=q.x*f.metric.x+q.y*f.metric.y;
        float y=q.x*f.metric.y+q.y*f.metric.z;
        if(x>=0.0 && x<=1.0 && y>=0.0 && y<=1.0)
            return true;
    }
    return false;
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
