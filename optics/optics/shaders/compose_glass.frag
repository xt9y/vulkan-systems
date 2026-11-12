#version 450
#extension GL_GOOGLE_include_directive : require
// Fast refractive glass resolve: no software BVH or per-light ray queries.
// Physically motivated Fresnel + bounded Snell-derived screen displacement.
// Deliberately avoids the image inversion of a fully refracting glass sphere.
layout(location=0) out vec4 color;
layout(set=0,binding=1) uniform sampler2D opaque_depth;
layout(set=0,binding=2) uniform sampler2D optic_normal;
layout(set=0,binding=3) uniform sampler2D optic_tint;
layout(set=0,binding=4) uniform sampler2D optic_material;
layout(set=0,binding=5) uniform sampler2D optic_depth;
layout(set=0,binding=16) uniform sampler2D scene_hdr;
layout(push_constant) uniform View {
    mat4 vp;
    vec4 eye,right,up,forward;
} camera;
#include "lighting_shared.glsl"
void main()
{
    ivec2 pixel=ivec2(gl_FragCoord.xy);
    vec4 mask=texelFetch(optic_normal,pixel,0);
    if(mask.a<.25 || mask.a>=.75)discard;
    float depth=texelFetch(optic_depth,pixel,0).r;
    if(depth>texelFetch(opaque_depth,pixel,0).r+.0003)discard;

    vec2 size=vec2(textureSize(scene_hdr,0));
    vec2 uv=gl_FragCoord.xy/size;
    vec4 tint=texelFetch(optic_tint,pixel,0);
    vec4 material=texelFetch(optic_material,pixel,0);
    vec3 normal=normalize(mask.rgb*2.0-1.0);
    vec2 screen=uv*2.0-1.0;
    vec3 incoming=normalize(camera.forward.xyz+
        camera.right.xyz*(screen.x*camera.right.w)-
        camera.up.xyz*(screen.y*camera.up.w));
    if(dot(normal,incoming)>0.0)normal=-normal;

    float ior=clamp(1.0+material.g,1.001,2.0);
    float cosine=clamp(dot(-incoming,normal),0.0,1.0);
    float f0=(ior-1.0)/(ior+1.0);
    f0*=f0;
    float edge=1.0-cosine;
    float fresnel=f0+(1.0-f0)*edge*edge*edge*edge*edge;
    bool solid=material.b>.5;

    vec2 shift=vec2(0.0);
    if(solid) {
        vec3 transmitted_ray=refract(incoming,normal,1.0/ior);
        vec3 bend=transmitted_ray-incoming;
        shift=vec2(
            dot(bend,camera.right.xyz)/max(camera.right.w,.001),
           -dot(bend,camera.up.xyz)/max(camera.up.w,.001))*.08;
        // Prevent image flips and enormous displacement at grazing angles.
        shift=clamp(shift,vec2(-.02),vec2(.02));
    }
    vec2 border=.5/size;
    vec3 background=texture(scene_hdr,clamp(uv+shift,border,1.0-border)).rgb;
    // A thin window is almost colorless. A solid receives modest volume
    // tint, not repeated face-by-face multiplication that turns it opaque.
    vec3 absorption=pow(clamp(tint.rgb,vec3(.05),vec3(1.0)),
                        vec3(solid?.16:.025));
    vec3 transmitted=background*absorption;
    float pass=clamp(tint.a,0.0,1.0);
    vec3 through=mix(tint.rgb,transmitted,pass);

    // Reuse only an offset scene sample for a faint grazing reflection.
    // No artificial bright blobs for point lights and no BVH traversal.
    vec3 reflected=texture(scene_hdr,
        clamp(uv-shift*.4,border,1.0-border)).rgb*.7;
    vec3 radiance=mix(through,reflected,fresnel);
    color=vec4(shared_tonemap(radiance,camera.forward.w),1.0);
}
