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
    if(mask.a<.25)discard;
    float surface_depth=texture(optic_depth,uv).r;
    float scene_depth=texture(opaque_depth,uv).r;
    if(surface_depth>scene_depth+.0003)discard;
    vec4 tint=texture(optic_tint,uv);
    vec4 optical=texture(optic_material,uv);
    float mattness=optical.r;
    float ior=1.0+optical.g; // R8G8B8A8 UNORM stores (ior - 1)
    bool solid_glass=optical.b>.5;
    vec3 N=normalize(mask.rgb*2.0-1.0);
    vec3 P=world_position(uv,surface_depth);
    vec3 incoming=normalize(P-camera.eye.xyz);
    if(dot(N,incoming)>0.0)N=-N;
    vec3 R=reflect(incoming,N);
    // Reflection rays return linear HDR. The renderer uses the same
    // exposure and monotonic tonemap curve for primary and reflected color.
    float strength=clamp(tint.a,0.0,1.0);
    bool glass=mask.a<.75;
#ifdef OPTICS_ONLY_MIRRORS
    if(camera.eye.w>1.5) {
        // Rasterized from the reflected camera, with horizontal sign
        // correction for the mirror's handedness. Same textures, shadow
        // BVH, PCloud GI, exposure and HDR transfer as the real scene.
        color=vec4(textureLod(planar_reflection,
            vec2(1.0-uv.x,uv.y),0.0).rgb*tint.rgb,1.0);
    } else {
        vec3 reflection=optics_perfect_mirror(P+N*.012,R);
        color=vec4(shared_tonemap(reflection*tint.rgb,camera.forward.w),1.0);
    }
#else
    // Sharp glass reflection uses the non-branching mirror tracer;
    // matte glass retains the existing glossy path.
    vec3 reflection=glass ?
        (mattness<=.35 ?
            optics_perfect_mirror(P+N*.03,R) :
            optics_offscreen_glossy(P+N*.03,R,mattness)) :
        optics_perfect_mirror(P+N*.012,R);
    float cosine=clamp(dot(-incoming,N),0.0,1.0);
    float f0=(ior-1.0)/(ior+1.0);
    f0*=f0;
    float fresnel=f0+(1.0-f0)*pow(1.0-cosine,5.0);
    if(glass) {
        // A solid cube/sphere transmits a refracted ray through its volume;
        // two interfaces cancel the unwanted single-surface lens artifact.
        // Thin panes keep one HDR fetch or a straight BVH ray.
        vec3 background=vec3(0.0);
        if(strength>.001) {
            if(solid_glass)
                background=optics_solid_transmission(
                    P,incoming,N,ior,clamp(tint.rgb,vec3(.001),vec3(1.0)));
            else
                background=camera.eye.w>2.5 ?
                    textureLod(scene_hdr,uv,0.0).rgb :
                    optics_offscreen_glossy(P+incoming*.03,incoming,mattness);
        }
        vec3 transmitted=solid_glass?background:
            background*mix(vec3(1.0),tint.rgb,.20);
        vec3 glass_color=mix(tint.rgb,transmitted,strength);
        vec3 radiance=mix(glass_color,reflection,fresnel)+
            optics_glass_specular(P,N,-incoming,mattness);
        color=vec4(shared_tonemap(radiance,camera.forward.w),1.0);
    } else {
        // Full-strength reflection through the editor's RGB color filter.
        color=vec4(shared_tonemap(reflection*tint.rgb,camera.forward.w),1.0);
    }
#endif
}
