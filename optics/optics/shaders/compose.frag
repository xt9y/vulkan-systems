#version 450
// Screen-visible reflections/refraction only. No ray-query, probe or
// cubemap fallback; rays ending outside the camera viewport produce no hit.
layout(location=0) in vec2 uv_in;
layout(location=0) out vec4 color;
layout(set=0,binding=0) uniform sampler2D lit_image;
layout(set=0,binding=1) uniform sampler2D opaque_depth;
layout(set=0,binding=2) uniform sampler2D optic_normal;
layout(set=0,binding=3) uniform sampler2D optic_tint;
layout(set=0,binding=4) uniform sampler2D optic_material;
layout(set=0,binding=5) uniform sampler2D optic_depth;
layout(std430,set=0,binding=6) readonly buffer CausticRadiance {
    uint caustic_values[];
};
layout(push_constant) uniform View {
    mat4 vp;
    vec4 eye, right, up, forward;
} camera;

vec3 world_position(vec2 uv,float depth)
{
    const float near_plane=.1,far_plane=1000.;
    float linear=near_plane*far_plane/
        max(far_plane-depth*(far_plane-near_plane),1e-5);
    vec2 screen=uv*2.0-1.0;
    return camera.eye.xyz+camera.forward.xyz*linear+
        camera.right.xyz*screen.x*linear*camera.right.w+
        camera.up.xyz*(-screen.y)*linear*camera.up.w;
}
bool trace_screen(vec3 origin,vec3 direction,out vec2 hit_uv)
{
    float t=.075;
    for(int i=0;i<32;++i)
    {
        t+=.06+t*.095;
        vec4 projected=camera.vp*vec4(origin+direction*t,1);
        if(projected.w<=.0001)break;
        vec3 ndc=projected.xyz/projected.w;
        vec2 coord=ndc.xy*.5+.5;
        if(any(lessThanEqual(coord,vec2(.001))) ||
           any(greaterThanEqual(coord,vec2(.999))) ||
           ndc.z<=0.0 || ndc.z>=1.0)break;
        float scene_d=texture(opaque_depth,coord).r;
        if(scene_d>=.999999)continue;
        // Depth intersection in view space; reject large disocclusions.
        vec3 scene_p=world_position(coord,scene_d);
        vec3 target=origin+direction*t;
        float along=dot(scene_p-origin,direction);
        float error=length(scene_p-target);
        if(along>0.0 && along<=t+.18 && error<max(.08,t*.065)) {
            hit_uv=coord;
            return true;
        }
    }
    return false;
}
vec3 sample_caustic(ivec2 pixel)
{
    ivec2 size=textureSize(lit_image,0);
    pixel=clamp(pixel,ivec2(0),size-1);
    uint offset=uint(pixel.x+pixel.y*size.x)*3u;
    return vec3(caustic_values[offset],caustic_values[offset+1u],
                caustic_values[offset+2u])*(1.0/256.0);
}
vec3 sample_rough(vec2 coord,float mattness)
{
    vec2 texel=1.0/vec2(textureSize(lit_image,0));
    float radius=mattness*mattness*12.;
    vec3 sum=texture(lit_image,coord).rgb*4.;
    sum+=texture(lit_image,coord+texel*vec2(radius,0)).rgb;
    sum+=texture(lit_image,coord-texel*vec2(radius,0)).rgb;
    sum+=texture(lit_image,coord+texel*vec2(0,radius)).rgb;
    sum+=texture(lit_image,coord-texel*vec2(0,radius)).rgb;
    return sum/8.;
}
void main()
{
    vec2 uv=gl_FragCoord.xy/vec2(textureSize(lit_image,0));
    ivec2 pixel=ivec2(gl_FragCoord.xy);
    vec3 caustic=sample_caustic(pixel);
    vec4 mask=texture(optic_normal,uv);
    if(mask.a<.25) {
        if(max(caustic.r,max(caustic.g,caustic.b))<=0.)discard;
        color=vec4(texture(lit_image,uv).rgb+caustic,1.);
        return;
    }
    float surface_depth=texture(optic_depth,uv).r;
    float scene_depth=texture(opaque_depth,uv).r;
    if(surface_depth>scene_depth+.0003) { // occluded optical material
        if(max(caustic.r,max(caustic.g,caustic.b))<=0.)discard;
        color=vec4(texture(lit_image,uv).rgb+caustic,1.);
        return;
    }
    vec4 tint=texture(optic_tint,uv);
    float mattness=texture(optic_material,uv).r;
    vec3 N=normalize(mask.rgb*2.0-1.0);
    vec3 P=world_position(uv,surface_depth);
    vec3 incoming=normalize(P-camera.eye.xyz);
    if(dot(N,incoming)>0.0)N=-N;
    vec3 R=reflect(incoming,N);
    vec2 ref_uv;
    bool reflected=trace_screen(P+N*.018,R,ref_uv);
    vec3 reflection=reflected?
        sample_rough(ref_uv,mattness)+
            sample_caustic(ivec2(ref_uv*vec2(textureSize(lit_image,0)))):vec3(0);
    float cosine=clamp(dot(-incoming,N),0.0,1.0);
    float fresnel=.04+.96*pow(1.0-cosine,5.0);
    if(mask.a<.75) {
        // Thin glass: bend one screen-visible camera ray toward the opaque
        // scene. The opaque image already excludes transparent surfaces.
        vec3 T=refract(incoming,N,1.0/1.52);
        vec2 transmit_uv;
        bool transmitted=length(T)>0.0 &&
                         trace_screen(P-N*.018,T,transmit_uv);
        vec3 transmission=transmitted ?
            sample_rough(transmit_uv,mattness)*mix(vec3(1),tint.rgb,.25) :
            texture(lit_image,uv).rgb;
        color=vec4(mix(transmission+caustic,reflection,fresnel),
                   clamp(tint.a,0.0,1.0));
    } else {
        // Reflective has no environmental fallback. The strength slider
        // blends only the result of an actual on-screen reflection query.
        color=vec4(reflection*tint.rgb,clamp(tint.a,0.0,1.0));
    }
}
