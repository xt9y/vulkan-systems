// Shared Cornell scene and GPU stackless BVH traversal.
// Stackless node layout is bvh::stackless_node from VulkanSystems.
#ifndef PHOTON_SCENE_GLSL
#define PHOTON_SCENE_GLSL
const float PM_PI=3.14159265358979323846;
const float PM_EPS=0.0008;
const float PM_SHADOW_MIN_T=0.00001;
const int PM_GRID=16;
// Exact previous-frame subpixel offset is required for correct
// temporal reprojection and stable hard geometry silhouettes.
vec2 pm_taa_jitter(uint frame)
{
    const vec2 pattern[16]=vec2[16](
        vec2(.5000000,.3333333),vec2(.2500000,.6666667),
        vec2(.7500000,.1111111),vec2(.1250000,.4444444),
        vec2(.6250000,.7777778),vec2(.3750000,.2222222),
        vec2(.8750000,.5555556),vec2(.0625000,.8888889),
        vec2(.5625000,.0370370),vec2(.3125000,.3703704),
        vec2(.8125000,.7037037),vec2(.1875000,.1481481),
        vec2(.6875000,.4814815),vec2(.4375000,.8148148),
        vec2(.9375000,.2592593),vec2(.0312500,.5925926)
    );
    return pattern[frame&15u];
}
struct Face { vec4 origin,u,v,normal,albedo,emission,metric; };
layout(std430,set=0,binding=1) readonly buffer Scene { Face faces[]; };
struct BVHNode { vec3 minimum; uint next; vec3 maximum; uint first_and_count; };
layout(std430,set=0,binding=9) readonly buffer Acceleration { BVHNode nodes[]; };
layout(std430,set=0,binding=10) readonly buffer Geometry { vec4 triangles[]; };
layout(push_constant) uniform Parameters {
    vec4 eye,right,up;
    uvec4 params;       // image dimensions, image history, maximum interactions
    uvec4 config;       // emitted photon paths, capacity, cache samples, flags
    vec4 settings;      // photon search radius
    vec4 sphere;        // sphere center xyz, radius w
    vec4 material;      // glass IOR, direct cache samples, atlas tile, shadow spp
} camera;
// Previous view is a separate SSBO: keeps push constants at 128 bytes,
// the Vulkan 1.1 portable guaranteed maximum.
layout(std430,set=0,binding=20) readonly buffer PreviousCamera {
    vec4 prev_eye,prev_right,prev_up;
};
layout(std430,set=0,binding=23) readonly buffer SceneInfo {
    uvec4 scene_desc; // face count, emitter, atlas columns, rows
    vec4 grid;        // grid origin xyz, cell size
    uvec4 extra;      // x: cutout count
};
layout(std430,set=0,binding=24) readonly buffer ReceiverCutouts {
    uvec4 cutouts[];  // receiver face, cover face
};
struct Hit { float t; int index; vec3 point; vec3 normal; bool front; };
struct Photon { vec4 position; vec4 power; uvec4 link; };
uint pm_random_bits(inout uint s)
{
    s^=s<<13;s^=s>>17;s^=s<<5;return s;
}
float pm_random(inout uint s)
{
    return float(pm_random_bits(s)>>8)*(1.0/16777216.0);
}
vec3 pm_hemisphere(vec3 n,inout uint seed)
{
    float r=sqrt(pm_random(seed));
    float phi=2.0*PM_PI*pm_random(seed);
    vec3 axis=abs(n.z)<0.99?vec3(0,0,1):vec3(0,1,0);
    vec3 t=normalize(cross(axis,n)),b=cross(n,t);
    return r*cos(phi)*t+r*sin(phi)*b+sqrt(max(0.0,1.0-r*r))*n;
}
bool pm_intersect_face(vec3 ro,vec3 rd,Face f,float limit,
                       float min_t,out float distance)
{
    float denom=dot(f.normal.xyz,rd);
    bool optical=f.albedo.w>0.0 || f.emission.w<0.0;
    if(optical ? abs(denom)<0.000001 : denom>=-0.000001)return false;
    float t=dot(f.origin.xyz-ro,f.normal.xyz)/denom;
    if(t<=min_t || t>=limit)return false;
    vec3 d=ro+rd*t-f.origin.xyz;
    vec2 q=vec2(dot(d,f.u.xyz),dot(d,f.v.xyz));
    float u=q.x*f.metric.x+q.y*f.metric.y;
    float v=q.x*f.metric.y+q.y*f.metric.z;
    if(u<0.0 || u>1.0 || v<0.0 || v>1.0 ||
       (f.normal.w>0.5 && u+v>1.0))return false;
    distance=t;
    return true;
}
bool pm_aabb(vec3 ro,vec3 rd,vec3 lo,vec3 hi,float limit,float min_t)
{
    vec3 safe=vec3(
        abs(rd.x)<1e-8?(rd.x<0.0?-1e-8:1e-8):rd.x,
        abs(rd.y)<1e-8?(rd.y<0.0?-1e-8:1e-8):rd.y,
        abs(rd.z)<1e-8?(rd.z<0.0?-1e-8:1e-8):rd.z);
    vec3 a=(lo-ro)/safe,b=(hi-ro)/safe;
    vec3 nearv=min(a,b),farv=max(a,b);
    float first=max(max(nearv.x,nearv.y),max(nearv.z,min_t));
    float last=min(min(farv.x,farv.y),min(farv.z,limit));
    return first<=last;
}
bool pm_closest(vec3 ro,vec3 rd,float limit,out Hit hit)
{
    hit.t=limit;hit.index=-1;hit.front=true;
    uint index=0u;
    // Use the exact node count supplied by the CPU-built VulkanBVH.
    for(uint steps=0u;index<camera.config.w && steps<camera.config.w*2u;++steps)
    {
        BVHNode n=nodes[index];
        if(!pm_aabb(ro,rd,n.minimum,n.maximum,hit.t,PM_EPS))
        {
            index=n.next;
            continue;
        }
        uint count=n.first_and_count&15u;
        if(count==0u){++index;continue;}
        uint first=n.first_and_count>>4u;
        for(uint j=0u;j<count;++j)
        {
            uint primitive=first+j;
            uint face_index=uint(triangles[primitive*3u].w);
            if(face_index>=scene_desc.x)continue;
            float distance;
            if(pm_intersect_face(ro,rd,faces[face_index],hit.t,PM_EPS,distance))
            {
                hit.t=distance;
                hit.index=int(face_index);
            }
        }
        index=n.next;
    }
    // Independently moving glass sphere is outside the static BVH.
    vec3 oc=ro-camera.sphere.xyz;
    float radius=camera.sphere.w;
    float b=dot(oc,rd),c=dot(oc,oc)-radius*radius;
    float delta=b*b-c;
    if(radius>0.0 && delta>=0.0)
    {
        float d=sqrt(delta);
        float t=-b-d;
        if(t<=PM_EPS)t=-b+d;
        if(t>PM_EPS && t<hit.t)
        {
            hit.t=t;
            hit.index=int(scene_desc.x);
        }
    }
    if(hit.index<0)return false;
    hit.point=ro+rd*hit.t;
    vec3 outward=hit.index==int(scene_desc.x)?
        normalize(hit.point-camera.sphere.xyz):faces[hit.index].normal.xyz;
    hit.front=dot(rd,outward)<0.0;
    hit.normal=hit.front?outward:-outward;
    return true;
}
// Occluded diffuse geometry blocks the light completely. The glass
// sphere instead transmits light (Schlick Fresnel through two surfaces).
// This is an inexpensive straight-line approximation: photon caustics
// handle focused illumination separately, without treating glass as an
// opaque shadow caster.
vec3 pm_surface_dielectric(vec3 direction,Hit hit,Face face,inout uint seed,
                           out bool reflected)
{
    // Fresnel and Snell law on arbitrary mesh faces, not a sphere shortcut.
    float ior=max(face.emission.w,1.001);
    float eta=hit.front?1.0/ior:ior;
    float cosine=clamp(dot(-direction,hit.normal),0.0,1.0);
    float r0=(ior-1.0)/(ior+1.0);
    float fresnel=r0*r0+(1.0-r0*r0)*pow(1.0-cosine,5.0);
    reflected=eta*eta*(1.0-cosine*cosine)>1.0 ||
        pm_random(seed)<fresnel;
    vec3 specular_direction=reflected ? reflect(direction,hit.normal) :
        refract(direction,hit.normal,eta);
    // Matte transmission spreads specular energy into a small cone.
    float mattness=clamp(face.metric.w,0.0,1.0);
    return normalize(mix(specular_direction,pm_hemisphere(normalize(specular_direction),seed),
                         mattness*mattness));
}
vec3 pm_surface_conductor(vec3 direction,Hit hit,Face face,inout uint seed)
{
    vec3 specular_direction=reflect(direction,hit.normal);
    float mattness=clamp(face.metric.w,0.0,1.0);
    return normalize(mix(specular_direction,pm_hemisphere(normalize(specular_direction),seed),
                         mattness*mattness));
}
// Occluded diffuse geometry blocks the light completely. The glass
// sphere instead transmits light (Schlick Fresnel through two surfaces).
// This is an inexpensive straight-line approximation: photon caustics
// handle focused illumination separately, without treating glass as an
// opaque shadow caster.
float pm_shadow_visibility(vec3 ro,vec3 rd,float limit)
{
    float visibility=1.0;
    uint index=0u;
    for(uint steps=0u;index<camera.config.w && steps<camera.config.w*2u;++steps)
    {
        BVHNode n=nodes[index];
        if(!pm_aabb(ro,rd,n.minimum,n.maximum,limit,PM_SHADOW_MIN_T))
        {
            index=n.next;
            continue;
        }
        uint count=n.first_and_count&15u;
        if(count==0u){++index;continue;}
        uint first=n.first_and_count>>4u;
        for(uint j=0u;j<count;++j)
        {
            uint face_id=uint(triangles[(first+j)*3u].w);
            if(face_id>=scene_desc.x)continue;
            float distance;
            if(pm_intersect_face(ro,rd,faces[face_id],limit,
                                 PM_SHADOW_MIN_T,distance)) {
                Face surface=faces[face_id];
                if(surface.albedo.w<=0.0)return 0.0;
                float ior=max(surface.emission.w,1.001);
                float f0=(ior-1.0)/(ior+1.0);
                f0*=f0;
                float c=abs(dot(rd,surface.normal.xyz));
                float fresnel=f0+(1.0-f0)*pow(1.0-c,5.0);
                visibility*=surface.albedo.w*(1.0-fresnel);
                if(visibility<0.0001)return 0.0;
            }
        }
        index=n.next;
    }
    float radius=camera.sphere.w;
    if(radius<=0.0)return visibility;
    vec3 delta=ro-camera.sphere.xyz;
    float b=dot(delta,rd);
    float determinant=b*b-dot(delta,delta)+radius*radius;
    if(determinant<=0.0)return visibility;
    float root=sqrt(determinant);
    float near_t=-b-root,far_t=-b+root;
    if(far_t<=PM_SHADOW_MIN_T || near_t>=limit)return visibility;
    float thickness=max(0.0,min(far_t,limit)-max(near_t,0.0));
    if(thickness<=0.0)return visibility;
    float cosine=clamp(root/radius,0.0,1.0);
    float ior=max(camera.material.x,1.001);
    float f0=(ior-1.0)/(ior+1.0);
    f0*=f0;
    float enter=f0+(1.0-f0)*pow(1.0-cosine,5.0);
    float eta=1.0/ior;
    float inside_cosine=sqrt(max(0.0,1.0-eta*eta*
                                 (1.0-cosine*cosine)));
    float leave=f0+(1.0-f0)*pow(1.0-inside_cosine,5.0);
    return visibility*(1.0-enter)*(1.0-leave);
}
vec3 pm_dielectric(vec3 direction,Hit hit,inout uint seed)
{
    float ior=max(camera.material.x,1.001);
    float eta=hit.front?1.0/ior:ior;
    float cosine=clamp(dot(-direction,hit.normal),0.0,1.0);
    float sin2=max(0.0,1.0-cosine*cosine);
    float r0=(ior-1.0)/(ior+1.0);
    r0*=r0;
    float fresnel=r0+(1.0-r0)*pow(1.0-cosine,5.0);
    if(eta*eta*sin2>1.0 || pm_random(seed)<fresnel)
        return normalize(reflect(direction,hit.normal));
    return normalize(refract(direction,hit.normal,eta));
}
uint pm_cell(vec3 p)
{
    ivec3 c=clamp(ivec3(floor((p-grid.xyz)/grid.w)),
                  ivec3(0),ivec3(PM_GRID-1));
    return uint(c.x+PM_GRID*(c.y+PM_GRID*c.z));
}
vec2 pm_face_uv(int i,vec3 p)
{
    Face f=faces[i];
    vec3 d=p-f.origin.xyz;
    vec2 q=vec2(dot(d,f.u.xyz),dot(d,f.v.xyz));
    return vec2(q.x*f.metric.x+q.y*f.metric.y,
                q.x*f.metric.y+q.y*f.metric.z);
}
// The Cornell objects cover parts of the original floor quad. Those
// atlas texels are not valid samples of the exposed receiver surface.
bool pm_receiver_masked(uint receiver,vec3 point)
{
    vec3 axis=faces[receiver].normal.xyz;
    for(uint i=0u;i<extra.x;++i)
    {
        uvec4 pair=cutouts[i];
        if(pair.x!=receiver || pair.y>=scene_desc.x)continue;
        Face cover=faces[pair.y];
        float cosine=dot(axis,cover.normal.xyz);
        if(abs(cosine)<1e-6)continue;
        float t=dot(cover.origin.xyz-point,cover.normal.xyz)/cosine;
        if(t<=PM_EPS)continue;
        vec2 uv=pm_face_uv(int(pair.y),point+axis*t);
        if(all(greaterThan(uv,vec2(0.0))) &&
           all(lessThan(uv,vec2(1.0))) &&
           (cover.normal.w<0.5 || uv.x+uv.y<1.0))return true;
    }
    return false;
}
vec3 pm_light_point(vec2 uv)
{
    Face source=faces[scene_desc.y];
    if(source.normal.w>0.5)
    {
        float root=sqrt(uv.x);
        uv=vec2(1.0-root,root*uv.y);
    }
    return source.origin.xyz+source.u.xyz*uv.x+source.v.xyz*uv.y;
}
#endif
