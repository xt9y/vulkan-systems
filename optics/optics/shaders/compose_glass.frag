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
layout(set=0,binding=17) uniform sampler2D planar_reflection;
layout(set=0,binding=18) uniform sampler2D second_normal;
layout(set=0,binding=19) uniform sampler2D second_tint;
layout(set=0,binding=20) uniform sampler2D second_material;
layout(set=0,binding=21) uniform sampler2D second_depth;
layout(set=0,binding=22) uniform sampler2D reflected_normal;
layout(set=0,binding=23) uniform sampler2D reflected_tint;
layout(set=0,binding=24) uniform sampler2D reflected_material;
layout(set=0,binding=25) uniform sampler2D reflected_depth;
layout(set=0,binding=26) uniform sampler2D reflected_opaque_depth;
layout(push_constant) uniform View {
    mat4 vp;
    vec4 eye,right,up,forward;
} camera;
#include "lighting_shared.glsl"
#include "glass_layer.glsl"
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

    // Screen-space refraction is deliberately bounded. A true two-surface
    // spherical lens can invert an image; this fast mode must not.
    vec2 shift=vec2(0.0);
    if(material.b>.5) {
        vec3 through=refract(incoming,normal,1.0/ior);
        if(dot(through,through)>1e-6) {
            vec3 bend=normalize(through)-incoming;
            shift=vec2(dot(bend,camera.right.xyz)/max(camera.right.w,.001),
                      -dot(bend,camera.up.xyz)/max(camera.up.w,.001))*.08;
            shift=clamp(shift,vec2(-.02),vec2(.02));
        }
    }
    vec2 border=.5/size;
    vec2 transmitted_uv=clamp(uv+shift,border,1.0-border);
    // Both ordinary geometry and planar mirrors must undergo the same
    // glass shading. Convert the normal HDR scene into the same display
    // space as the already-tonemapped planar reflection, then substitute
    // only the background source if the refracted view hits the mirror.
    vec3 background_ldr=shared_tonemap(
        texture(scene_hdr,transmitted_uv).rgb,camera.forward.w);
    float pass=clamp(tint.a,0.0,1.0);
    float mirror_surface_z=1e30;
    vec3 reflection_ldr=shared_tonemap(
        texture(scene_hdr,clamp(uv-shift*.4,border,1.0-border)).rgb*.7,
        camera.forward.w);

    // The regular planar mirror resolves from the *tonemapped LDR*
    // reflection at (1-uv.x,uv.y) and applies the mirror's color tint.
    // Use that exact same source here. Reconstructing a reflected-camera
    // projection or sampling its HDR intermediate does not match the
    // image actually displayed by the mirror.
    if(camera.vp[1].w>.5) {
        vec4 plane=camera.vp[0];
        vec3 N=plane.xyz;
        vec2 mirror_uv=transmitted_uv;
        vec2 ndc=mirror_uv*2.0-1.0;
        vec3 ray=normalize(camera.forward.xyz+
            camera.right.xyz*(ndc.x*camera.right.w)-
            camera.up.xyz*(ndc.y*camera.up.w));
        float denom=dot(N,ray);
        if(abs(denom)>1e-5) {
            float t=-(dot(N,camera.eye.xyz)+plane.w)/denom;
            float mirror_z=t*dot(ray,camera.forward.xyz);
            // Depth uses the same 0.1..1 viewport mapping as draw_target.
            float glass_ndc=clamp((depth-.1)/.9,0.0,1.0);
            float glass_z=100.0/max(1000.0-999.9*glass_ndc,1e-5);
            if(t>0.0 && mirror_z>glass_z+.02) {
                vec3 hit=camera.eye.xyz+ray*t;
                vec3 rel=hit-camera.vp[1].xyz;
                vec3 axis_y=normalize(camera.vp[2].xyz);
                vec3 axis_z=normalize(cross(N,axis_y));
                if(abs(dot(rel,axis_y))<=camera.vp[2].w &&
                   abs(dot(rel,axis_z))<=camera.vp[3].w) {
                    // Only use the mirror when no opaque object blocks
                    // it from this refracted screen-space direction.
                    float background_depth=texture(opaque_depth,mirror_uv).r;
                    float opaque_ndc=clamp((background_depth-.1)/.9,0.0,1.0);
                    float opaque_z=100.0/max(1000.0-999.9*opaque_ndc,1e-5);
                    if(opaque_z+.02>=mirror_z) {
                        vec3 mirror_ldr=texture(planar_reflection,
                            vec2(1.0-mirror_uv.x,mirror_uv.y)).rgb*
                            clamp(camera.vp[3].rgb,vec3(0.0),vec3(1.0));
                        vec2 reflected_uv=vec2(
                            1.0-mirror_uv.x,mirror_uv.y);
                        // The mirror's reflected-camera G-buffer captures
                        // glass that is invisible to the original view.
                        vec4 back_mask=texture(
                            reflected_normal,reflected_uv);
                        float back_depth=texture(
                            reflected_depth,reflected_uv).r;
                        float back_opaque=texture(
                            reflected_opaque_depth,reflected_uv).r;
                        if(back_mask.a>.25 && back_mask.a<.75 &&
                           back_depth<back_opaque+.0003) {
                            vec4 back_tint=texture(
                                reflected_tint,reflected_uv);
                            vec4 back_material=texture(
                                reflected_material,reflected_uv);
                            mirror_ldr=optics_glass_layer(
                                mirror_ldr,back_tint.rgb,back_tint.a,
                                back_material.b);
                        }
                        background_ldr=mirror_ldr;
                        mirror_surface_z=mirror_z;
                    }
                }
            }
        }
    }
    // Sample the second FRONT-facing glass layer behind this one. Its
    // own depth test guarantees the layer belongs behind the first
    // optical surface; opaque geometry still wins if nearer.
    vec4 second_mask=texture(second_normal,transmitted_uv);
    float second_z=texture(second_depth,transmitted_uv).r;
    float opaque_z=texture(opaque_depth,transmitted_uv).r;
    float second_ndc=clamp((second_z-.1)/.9,0.0,1.0);
    float second_linear=100.0/
        max(1000.0-999.9*second_ndc,1e-5);
    if(second_mask.a>.25 && second_mask.a<.75 &&
       second_z>depth+.0001 && second_z<opaque_z+.0003 &&
       second_linear+.02<mirror_surface_z) {
        vec4 behind_tint=texture(second_tint,transmitted_uv);
        vec4 behind_material=texture(second_material,transmitted_uv);
        background_ldr=optics_glass_layer(
            background_ldr,behind_tint.rgb,behind_tint.a,
            behind_material.b);
    }
    vec3 glass=optics_glass_layer(background_ldr,tint.rgb,pass,
                                 solid?1.0:0.0);
    color=vec4(mix(glass,reflection_ldr,fresnel),1.0);
}
