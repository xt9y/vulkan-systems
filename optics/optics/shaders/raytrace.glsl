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
// A face maps its three source UVs onto a subrectangle of the packed
// RGBA8 atlas. Face indices match the Photon Mapping BVH exactly.
struct OpticsUV {
    vec4 uv01;
    vec4 uv2_origin;
    vec4 rect;
};
layout(std430,set=0,binding=13) readonly buffer OpticalUV {
    OpticsUV optics_uv[];
};
layout(set=0,binding=14) uniform sampler2D optics_atlas;
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
    OpticsUV mapping=optics_uv[hit.face_id];
    if(mapping.rect.z>.5) {
        vec3 d=hit.position-f.origin.xyz;
        vec2 q=vec2(dot(d,f.u.xyz),dot(d,f.v.xyz));
        vec2 bary=vec2(q.x*f.metric.x+q.y*f.metric.y,
                       q.x*f.metric.y+q.y*f.metric.z);
        vec2 uv=fract(mapping.uv01.xy*(1.0-bary.x-bary.y)+
                      mapping.uv01.zw*bary.x+
                      mapping.uv2_origin.xy*bary.y);
        vec2 atlas_size=vec2(textureSize(optics_atlas,0));
        vec2 atlas_min=mapping.uv2_origin.zw+0.5/atlas_size;
        vec2 atlas_max=mapping.uv2_origin.zw+mapping.rect.xy-0.5/atlas_size;
        vec2 atlas_uv=mix(atlas_min,atlas_max,uv);
        // Explicit LOD keeps sampling defined in divergent BVH ray paths.
        albedo*=pow(max(textureLod(optics_atlas,atlas_uv,0.0).rgb,
                        vec3(0.0)),vec3(2.2));
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
        if(surface.albedo.w>.001) {
            float strength=clamp(surface.albedo.w,0.0,1.0);
            float ior=max(1.001,surface.emission.w);
            bool entering=dot(path.direction,surface.normal.xyz)<0.0;
            float eta=entering?1.0/ior:ior;
            vec3 normal=hit.normal;
            float cosine=clamp(dot(-path.direction,normal),0.0,1.0);
            float r0=(ior-1.0)/(ior+1.0);
            r0*=r0;
            float fresnel=r0+(1.0-r0)*pow(1.0-cosine,5.0);
            vec3 T=refract(path.direction,normal,eta);
            bool total_internal=dot(T,T)<.00001;
            if(total_internal)fresnel=1.0;
            vec3 tint=max(surface.albedo.rgb,vec3(0.0));
            // Low pass-through still reveals the material itself and
            // keeps the edge visible in nested reflections/refractions.
            if(entering)
                accumulated+=path.throughput*tint*(1.0-strength)*
                             (1.0-fresnel);
            vec3 reflected=path.throughput*fresnel;
            if(queued<16 && max(reflected.r,max(reflected.g,reflected.b))>.003)
                pending[queued++]=OpticsPath(
                    hit.position+normal*.012,
                    normalize(reflect(path.direction,normal)),
                    reflected,path.depth+1);
            if(!total_internal && strength>.003) {
                vec3 transmitted=path.throughput*(1.0-fresnel)*strength;
                if(entering)transmitted*=tint;
                if(queued<16)
                    pending[queued++]=OpticsPath(
                        hit.position-normal*.012,normalize(T),
                        transmitted,path.depth+1);
            }
            continue;
        }
        if(surface.emission.w<-.001) {
            float reflectivity=clamp(-surface.emission.w,0.0,1.0);
            vec3 tint=max(surface.albedo.rgb,vec3(0.0));
            accumulated+=path.throughput*tint*.10*(1.0-reflectivity);
            path.throughput*=tint*reflectivity;
            path.origin=hit.position+hit.normal*.012;
            path.direction=normalize(reflect(path.direction,hit.normal));
            path.depth++;
            if(queued<16)pending[queued++]=path;
            continue;
        }
        // Reuse the *exact* lit raster color when this hit is visible
        // in the primary camera, preserving textures and animation.
        vec4 projected=camera.vp*vec4(hit.position,1.0);
        bool from_screen=false;
        if(projected.w>.00001) {
            vec3 ndc=projected.xyz/projected.w;
            vec2 screen=ndc.xy*.5+.5;
            if(all(greaterThanEqual(screen,vec2(.001))) &&
               all(lessThanEqual(screen,vec2(.999))) &&
               ndc.z>=0.0 && ndc.z<=1.0) {
                float expected_depth=.1+.9*ndc.z;
                float actual_depth=textureLod(opaque_depth,screen,0.0).r;
                // Exclude near-depth first-person weapon pixels.
                if(actual_depth>=.1 &&
                   abs(actual_depth-expected_depth)<.0004) {
                    accumulated+=path.throughput*
                        textureLod(lit_image,screen,0.0).rgb;
                    from_screen=true;
                }
            }
        }
        if(!from_screen) {
            vec3 radiance=optics_surface_radiance(hit,path.direction);
            // Match the main scene's output transfer for off-screen hits.
            accumulated+=path.throughput*
                (radiance/(vec3(1.0)+max(radiance,vec3(0.0))));
        }
    }
    return accumulated;
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
