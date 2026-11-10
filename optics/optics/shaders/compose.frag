#version 450
#extension GL_GOOGLE_include_directive : require
// Opaque/reflective material resolve. Every reflective pixel traces the
// same GPU BVH as VulkanPMap, including geometry outside the viewport.
// Closed glass geometry refracts at both entry and exit; all optical
// radiance is combined in linear HDR before the PCloud tone curve.
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
void main()
{
    vec2 uv=gl_FragCoord.xy/vec2(textureSize(lit_image,0));
    vec4 mask=texture(optic_normal,uv);
    if(mask.a<.25)discard;
    float surface_depth=texture(optic_depth,uv).r;
    float scene_depth=texture(opaque_depth,uv).r;
    if(surface_depth>scene_depth+.0003)discard;
    // OPTICS_DEBUG_MASK=1: cyan glass / magenta mirrors means the real
    // raster mask and its depth test reached the optical compositor.
    if(camera.eye.w>1.5) {
        color=vec4(mask.a<.75?vec3(.05,1.0,1.0):
                                vec3(1.0,.05,1.0),1.0);
        return;
    }
    vec4 tint=texture(optic_tint,uv);
    float mattness=texture(optic_material,uv).r;
    vec3 N=normalize(mask.rgb*2.0-1.0);
    vec3 P=world_position(uv,surface_depth);
    vec3 incoming=normalize(P-camera.eye.xyz);
    if(dot(N,incoming)>0.0)N=-N;
    vec3 R=reflect(incoming,N);
    // Linear-HDR material composition. PCloud performs exactly this
    // exposure/tonemap on its opaque pixels; optics does it once here.
    float strength=clamp(tint.a,0.0,1.0);
    float cosine=clamp(dot(-incoming,N),0.0,1.0);
    vec3 radiance=vec3(0.0);
    if(mask.a<.75) {
        float ior=max(texture(optic_material,uv).g,1.001);
        float fresnel=optics_dielectric_fresnel(cosine,1.0,ior);
        vec3 reflected=optics_offscreen_glossy(P+N*.012,R,mattness,
                                               1.0,vec3(1.0));
        vec3 transmitted=refract(incoming,N,1.0/ior);
        vec3 tint_filter=clamp(tint.rgb,vec3(.001),vec3(1.0));
        if(dot(transmitted,transmitted)>1e-8 && strength>.003) {
            // This ray starts *inside* the solid and finds the actual exit
            // boundary using the shared scene BVH. No screen-space overlay.
            vec3 extinction=mix(vec3(1.0),tint_filter,strength)*strength;
            vec3 behind=optics_offscreen_glossy(
                P-N*.012,normalize(transmitted),mattness,ior,extinction);
            radiance+=behind*(1.0-fresnel);
        }
        radiance+=reflected*fresnel;
        // Light sources have zero geometric area in the server scene.
        // Their GGX interface reflections must therefore be evaluated
        // analytically: BVH rays cannot intersect an infinitesimal light.
        float f0=(ior-1.0)/(ior+1.0);
        radiance+=optics_interface_lighting(
            P,N,normalize(-incoming),vec3(f0*f0),mattness);
    } else {
        // Conductor Fresnel: RGB F0 is the mirror's tinted reflectivity.
        // One initial ray for sharp mirrors; only rough ones get extras.
        vec3 f0=clamp(tint.rgb*strength,vec3(0.0),vec3(1.0));
        vec3 fresnel=f0+(vec3(1.0)-f0)*pow(1.0-cosine,5.0);
        vec3 reflected=optics_offscreen_glossy(P+N*.012,R,mattness,
                                               1.0,vec3(1.0));
        radiance=reflected*fresnel+tint.rgb*.10*(1.0-strength);
        radiance+=optics_interface_lighting(
            P,N,normalize(-incoming),f0,mattness);
    }
    radiance=max(radiance,vec3(0.0))*max(camera.forward.w,0.0);
    color=vec4(radiance/(vec3(1.0)+radiance),1.0);
}
