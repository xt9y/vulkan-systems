// Opaque/reflective material resolve. Every reflective pixel traces the
// optics-owned BVH, including geometry outside the viewport.
// Single-glass transmission reuses HDR; nested optics retain BVH rays.
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
layout(set=0,binding=16) uniform sampler2D scene_hdr;
layout(set=0,binding=17) uniform sampler2D planar_reflection;

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
void main()
{
    vec2 uv=gl_FragCoord.xy/vec2(textureSize(lit_image,0));
    vec4 mask=texture(optic_normal,uv);
    // The separate cheap glass shader resolves every dielectric fragment.
    // Only reflective surfaces run the expensive BVH mirror traversal.
    if(mask.a<.75)discard;
    float depth=texture(optic_depth,uv).r;
    if(depth>texture(opaque_depth,uv).r+.0003)discard;
    vec3 N=normalize(mask.rgb*2.0-1.0);
    vec3 P=world_position(uv,depth);
    vec3 incoming=normalize(P-camera.eye.xyz);
    if(dot(N,incoming)>0.0)N=-N;
    vec4 tint=texture(optic_tint,uv);
    vec3 R=reflect(incoming,N);
#ifdef OPTICS_ONLY_MIRRORS
    if(camera.eye.w>1.5 && camera.eye.w<2.5)
        color=vec4(textureLod(planar_reflection,
            vec2(1.0-uv.x,uv.y),0.0).rgb*tint.rgb,1.0);
    else
#endif
        color=vec4(shared_tonemap(
            optics_perfect_mirror(P+N*.012,R)*tint.rgb,
            camera.forward.w),1.0);
}
