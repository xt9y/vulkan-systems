// Reusable software-BVH reflection queries over Photon Mapping's scene
// buffers. BVH leaf ordering and the 112-byte face layout match VulkanPMap.
// Full-resolution reflected and refracted rays use the same triangle BVH
// as Photon Mapping. Per-hit shadows, PBR material parameters, photons,
// textures and smooth normals avoid any low-resolution reflection cache.
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
layout(set=0,binding=16) uniform sampler2D hdr_image;
// Per-face vertex normals share the exact face ordering of the traced BVH.
// They do not change triangle intersections; curved meshes keep smooth
// reflected directions instead of showing their individual polygons.
struct OpticsNormals {vec4 a,b,c;};
layout(std430,set=0,binding=15) readonly buffer OpticalNormals {
    OpticsNormals optics_normals[];
};
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
bool optics_aabb(vec3 ro,vec3 safe,vec3 lo,vec3 hi,float max_t)
{
    vec3 lo_t=(lo-ro)/safe;
    vec3 hi_t=(hi-ro)/safe;
    vec3 minimum=min(lo_t,hi_t),maximum=max(lo_t,hi_t);
    float near_t=max(max(minimum.x,minimum.y),max(minimum.z,.025));
    float far_t=min(min(maximum.x,maximum.y),min(maximum.z,max_t));
    return near_t<=far_t;
}
vec2 optics_atlas_coord(OpticsUV mapping,vec2 bary)
{
    vec2 uv=fract(mapping.uv01.xy*(1.0-bary.x-bary.y)+
                  mapping.uv01.zw*bary.x+
                  mapping.uv2_origin.xy*bary.y);
    vec2 size=vec2(textureSize(optics_atlas,0));
    vec2 lo=mapping.uv2_origin.zw+.5/size;
    vec2 hi=mapping.uv2_origin.zw+mapping.rect.xy-.5/size;
    return mix(lo,hi,uv);
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
    if(mapping.rect.z>.5 && mapping.rect.w>.0) {
        vec2 address=optics_atlas_coord(mapping,vec2(u,v));
        if(textureLod(optics_atlas,address,0.0).a<mapping.rect.w)
            return false;
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
    vec3 safe=optics_safe_direction(rd);
    for(uint steps=0u;index<total_nodes && steps<total_nodes*2u;++steps) {
        OpticsNode node=optics_nodes[index];
        if(!optics_aabb(ro,safe,node.minimum,node.maximum,result.t)) {
            index=node.next;
            continue;
        }
        uint count=node.first_and_count&15u;
        if(count==0u) {++index;continue;}
        uint first=node.first_and_count>>4u;
        for(uint k=0u;k<count;++k) {
            uint face_id=uint(optics_triangles[(first+k)*3u].w);
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
    vec3 safe=optics_safe_direction(rd);
    for(uint step=0u;index<total && step<total*2u;++step) {
        OpticsNode node=optics_nodes[index];
        if(!optics_aabb(ro,safe,node.minimum,node.maximum,limit)) {
            index=node.next;
            continue;
        }
        uint count=node.first_and_count&15u;
        if(count==0u){++index;continue;}
        uint first=node.first_and_count>>4u;
        for(uint k=0u;k<count;++k) {
            uint id=uint(optics_triangles[(first+k)*3u].w);
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
// Same direct-light GGX/Smith/Schlick BRDF as PCloud light.comp,
// evaluated only at secondary surfaces outside the primary screen.
vec3 optics_brdf(vec3 albedo,float metallic,float roughness,
                 vec3 N,vec3 V,vec3 L)
{
    float nl=max(dot(N,L),0.0);
    if(nl<=0.0)return vec3(0.0);
    vec3 H=normalize(V+L);
    float nv=max(dot(N,V),1e-4);
    float nh=max(dot(N,H),0.0);
    float vh=max(dot(V,H),0.0);
    float a=roughness*roughness,a2=a*a;
    float d=nh*nh*(a2-1.0)+1.0;
    float D=a2/max(OPTICS_PI*d*d,1e-6);
    float k=(roughness+1.0)*(roughness+1.0)*.125;
    float G=(nv/(nv*(1.0-k)+k))*(nl/(nl*(1.0-k)+k));
    vec3 F0=mix(vec3(.04),albedo,metallic);
    float m=1.0-vh,m2=m*m;
    vec3 F=F0+(vec3(1.0)-F0)*m2*m2*m;
    return ((vec3(1.0)-F)*(1.0-metallic)*albedo/OPTICS_PI+
            D*G*F/max(4.0*nv*nl,1e-4))*nl;
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
        vec2 atlas_uv=optics_atlas_coord(mapping,bary);
        albedo*=pow(max(textureLod(optics_atlas,atlas_uv,0.0).rgb,
                        vec3(0.0)),vec3(2.2));
    }
    OpticsNormals material=optics_normals[hit.face_id];
    float metallic=clamp(material.a.w,0.0,1.0);
    float roughness=clamp(material.b.w,.045,1.0);
    float ao=clamp(material.c.w,0.0,1.0);
    vec3 radiance=max(f.emission.rgb,vec3(0.0))+
                  optics_photon_radiance(hit,albedo)*(1.0-metallic)*ao;
    vec3 normal=hit.normal;
    vec3 view_direction=normalize(-rd);
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
        // Early-out any-opaque traversal is exact for straight-line
        // point-light visibility, including opaque surfaces behind glass.
        if(optics_occluded(hit.position+hit.geometric_normal*.03,L,
                           distance-.06,hit.face_id))continue;
        float q=clamp(distance/radius,0.0,1.0);
        float attenuation=1.0-q*q*(3.0-2.0*q);
        radiance+=optics_brdf(albedo,metallic,roughness,
                             normal,view_direction,L)*
                  light.color_strength.rgb*
                  (light.color_strength.w*attenuation);
    }
    return radiance;
}
// Unpolarized Fresnel for the actual incident and transmitted IORs.
// Unlike Schlick, this also handles total internal reflection exactly.
float optics_dielectric_fresnel(float cosine,float eta_i,float eta_t)
{
    cosine=clamp(cosine,0.0,1.0);
    float eta=eta_i/eta_t;
    float transmitted_sine2=eta*eta*(1.0-cosine*cosine);
    if(transmitted_sine2>=1.0)return 1.0;
    float transmitted_cosine=sqrt(max(0.0,1.0-transmitted_sine2));
    float rs=(eta_i*cosine-eta_t*transmitted_cosine)/
             max(eta_i*cosine+eta_t*transmitted_cosine,1e-6);
    float rp=(eta_t*cosine-eta_i*transmitted_cosine)/
             max(eta_t*cosine+eta_i*transmitted_cosine,1e-6);
    return clamp(.5*(rs*rs+rp*rp),0.0,1.0);
}
vec3 optics_absorption(vec3 transmittance,float travel)
{
    // Tint is the transmission over one scene metre. Beer-Lambert
    // yields white short paths and naturally deeper colors in thick glass.
    return pow(clamp(transmittance,vec3(.001),vec3(1.0)),
               vec3(max(0.0,travel)));
}
struct OpticsPath {
    vec3 origin;
    vec3 direction;
    vec3 throughput;
    int depth;
    float medium_ior;
    vec3 medium_tint;
};
vec3 optics_offscreen_radiance(vec3 origin,vec3 direction,
                                float medium_ior,vec3 medium_tint)
{
    if(optics_desc.x==0u)return vec3(0.0);
    OpticsPath pending[16];
    int queued=1;
    pending[0]=OpticsPath(origin,normalize(direction),vec3(1.0),0,
                           medium_ior,medium_tint);
    vec3 accumulated=vec3(0.0);
    // Queue both physical Fresnel branches. No fake scene/environment
    // substitutions and no recursion-specific shader compilation.
    for(int iteration=0;iteration<64 && queued>0;++iteration) {
        OpticsPath path=pending[--queued];
        if(path.depth>=10 ||
           max(path.throughput.r,max(path.throughput.g,
               path.throughput.b))<.003)continue;
        OpticsHit hit;
        if(!optics_closest(path.origin,path.direction,1000.0,hit))
            continue;
        if(path.medium_ior>1.001)
            path.throughput*=optics_absorption(path.medium_tint,hit.t);
        OpticsFace surface=optics_faces[hit.face_id];
        if(surface.emission.w>1.0) {
            float strength=clamp(surface.albedo.w,0.0,1.0);
            float ior=clamp(surface.emission.w,1.001,3.0);
            // Mesh outward orientation determines entry/exit, independently
            // of the normal flipped toward the ray for shading.
            bool entering=dot(path.direction,surface.normal.xyz)<0.0;
            float eta_i=path.medium_ior;
            float eta_t=entering?ior:1.0;
            vec3 normal=hit.normal;
            float cosine=clamp(dot(-path.direction,normal),0.0,1.0);
            float fresnel=optics_dielectric_fresnel(cosine,eta_i,eta_t);
            vec3 tint=clamp(surface.albedo.rgb,vec3(.001),vec3(1.0));
            // Reflection remains in the incident medium.
            vec3 reflected=path.throughput*fresnel;
            if(queued<16 && max(reflected.r,max(reflected.g,reflected.b))>.003)
                pending[queued++]=OpticsPath(
                    hit.position+hit.geometric_normal*.012,
                    normalize(reflect(path.direction,normal)),
                    reflected,path.depth+1,path.medium_ior,path.medium_tint);
            vec3 transmission=refract(path.direction,normal,eta_i/eta_t);
            if(dot(transmission,transmission)>1e-8 && strength>.003) {
                vec3 transmitted=path.throughput*(1.0-fresnel)*strength;
                if(queued<16)
                    pending[queued++]=OpticsPath(
                        hit.position-hit.geometric_normal*.012,
                        normalize(transmission),transmitted,path.depth+1,
                        eta_t,entering?tint:vec3(1.0));
            }
            if(entering)
                accumulated+=path.throughput*tint*(1.0-strength)*
                             (1.0-fresnel);
            continue;
        }
        if(surface.emission.w<-.001) {
            float reflectivity=clamp(-surface.emission.w,0.0,1.0);
            vec3 tint=clamp(surface.albedo.rgb,vec3(0.0),vec3(1.0));
            float cosine=clamp(dot(-path.direction,hit.normal),0.0,1.0);
            vec3 f0=tint*reflectivity;
            vec3 f=f0+(vec3(1.0)-f0)*pow(1.0-cosine,5.0);
            accumulated+=path.throughput*tint*.10*(1.0-reflectivity);
            path.throughput*=f;
            path.origin=hit.position+hit.geometric_normal*.012;
            path.direction=normalize(reflect(path.direction,hit.normal));
            path.depth++;
            if(queued<16)pending[queued++]=path;
            continue;
        }
        // If the same point is raster-visible, share the exact HDR
        // lighting including its PBR BRDF, animated textures and emission.
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
                if(actual_depth>=.1 &&
                   abs(actual_depth-expected_depth)<.0004) {
                    accumulated+=path.throughput*
                        textureLod(hdr_image,screen,0.0).rgb;
                    from_screen=true;
                }
            }
        }
        if(!from_screen)
            accumulated+=path.throughput*
                         optics_surface_radiance(hit,path.direction);
    }
    return accumulated;
}
vec3 optics_offscreen_glossy(vec3 origin,vec3 direction,float mattness,
                            float medium_ior,vec3 medium_tint)
{
    vec3 center=optics_offscreen_radiance(origin,direction,
                                          medium_ior,medium_tint);
    // Deterministic cone sampling only for matte, off-screen reflections.
    // No noise, stochastic temporal filter or environment fallback.
    if(mattness<=.08)return center;
    vec3 axis=abs(direction.y)<.9?vec3(0,1,0):vec3(1,0,0);
    vec3 tangent=normalize(cross(direction,axis));
    vec3 bitangent=normalize(cross(direction,tangent));
    float spread=mattness*mattness*.16;
    vec3 offset=(tangent+bitangent)*spread;
    vec3 a=optics_offscreen_radiance(origin,normalize(direction+offset),
                                    medium_ior,medium_tint);
    vec3 b=optics_offscreen_radiance(origin,normalize(direction-offset),
                                    medium_ior,medium_tint);
    return (center*2.0+a+b)*.25;
}
#endif
