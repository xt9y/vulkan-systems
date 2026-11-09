#version 450
#extension GL_GOOGLE_include_directive : require
// Opaque/reflective material resolve. Every reflective pixel traces the
// same GPU BVH as VulkanPMap, including geometry outside the viewport.
// Glass transmission still samples the opaque image when possible.
layout(location=0) in vec2 uv_in;
layout(location=0) out vec4 color;
layout(set=0,binding=0) uniform sampler2D lit_image;
layout(set=0,binding=1) uniform sampler2D opaque_depth;
layout(set=0,binding=2) uniform sampler2D optic_normal;
layout(set=0,binding=3) uniform sampler2D optic_tint;
layout(set=0,binding=4) uniform sampler2D optic_material;
layout(set=0,binding=5) uniform sampler2D optic_depth;
layout(push_constant) uniform View {
    mat4 vp;
    vec4 eye, right, up, forward;
} camera;
#include "raytrace.glsl"

vec3 world_position(vec2 uv,float depth)
{
    const float near_plane=.1,far_plane=1000.;
    // VulkanGame renders geometry with viewport minDepth=0.1,maxDepth=1.
    // The depth attachment contains window depth, NOT projection NDC depth.
    float ndc_depth=clamp((depth-.1)/.9,0.0,1.0);
    float linear=near_plane*far_plane/
        max(far_plane-ndc_depth*(far_plane-near_plane),1e-5);
    vec2 screen=uv*2.0-1.0;
    return camera.eye.xyz+camera.forward.xyz*linear+
        camera.right.xyz*screen.x*linear*camera.right.w+
        camera.up.xyz*(-screen.y)*linear*camera.up.w;
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
    vec4 mask=texture(optic_normal,uv);
    if(mask.a<.25)discard;
    float surface_depth=texture(optic_depth,uv).r;
    float scene_depth=texture(opaque_depth,uv).r;
    if(surface_depth>scene_depth+.0003)discard;
    vec4 tint=texture(optic_tint,uv);
    float mattness=texture(optic_material,uv).r;
    vec3 N=normalize(mask.rgb*2.0-1.0);
    vec3 P=world_position(uv,surface_depth);
    vec3 incoming=normalize(P-camera.eye.xyz);
    if(dot(N,incoming)>0.0)N=-N;
    vec3 R=reflect(incoming,N);
    // Every mirror pixel resolves a physical ray against the *same* GPU
    // scene BVH: no screen-depth jumps, disocclusion layering or probes.
    vec3 reflection=optics_offscreen_glossy(P+N*.03,R,mattness);
    float cosine=clamp(dot(-incoming,N),0.0,1.0);
    float fresnel=.04+.96*pow(1.0-cosine,5.0);
    if(mask.a<.75) {
        // A thin dielectric samples the rasterized opaque background in
        // screen space. This is the primary transmission path, not an
        // off-screen fallback; all geometry outside the viewport is omitted.
        const float ior=1.52;
        vec3 T=refract(incoming,N,1.0/ior);
        vec4 refracted_clip=camera.vp*vec4(P+T*.30,1.0);
        vec2 transmit_uv=refracted_clip.xy/max(refracted_clip.w,.00001)*.5+.5;
        bool transmitted=refracted_clip.w>0.0 &&
            all(greaterThanEqual(transmit_uv,vec2(0.0))) &&
            all(lessThan(transmit_uv,vec2(1.0))) &&
            dot(T,T)>.01;
        vec3 background=transmitted ?
            sample_rough(transmit_uv,mattness) :
            optics_offscreen_glossy(P-N*.03,T,mattness);
        // At zero pass-through show opaque tinted glass, not an invisible
        // hole. At one show the refracted image, including receiver caustics.
        float transmission=clamp(tint.a,0.0,1.0);
        vec3 glass=mix(tint.rgb,background*mix(vec3(1.0),tint.rgb,.20),
                       transmission);
        color=vec4(mix(glass,reflection,fresnel),1.0);
    } else {
        // Mirrors only show traced on-screen hits, never probes/cubemaps.
        float strength=clamp(tint.a,0.0,1.0);
        // Reflectivity is a material parameter, not alpha blending.
        color=vec4(mix(tint.rgb*.10,reflection*tint.rgb,strength),1.0);
    }
}
