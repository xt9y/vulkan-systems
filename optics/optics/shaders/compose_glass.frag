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
    vec3 background=texture(scene_hdr,transmitted_uv).rgb;
    vec3 absorption=pow(clamp(tint.rgb,vec3(.05),vec3(1.0)),
                        vec3(material.b>.5?.04:.01));
    float pass=clamp(tint.a,0.0,1.0);
    vec3 transmission=mix(tint.rgb,background*absorption,pass);
    vec3 reflected=texture(scene_hdr,
        clamp(uv-shift*.4,border,1.0-border)).rgb*.7;
    vec3 radiance=mix(transmission,reflected,fresnel);
    vec3 glass_ldr=shared_tonemap(radiance,camera.forward.w);

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
                        // The transmitted mirror is not the final
                        // glass color. Keep the same glass layer already
                        // computed for the ordinary world visible, even
                        // when pass-through is nearly 100 percent.
                        // A darker authored RGB pigment makes this layer
                        // denser; clear glass stays more transparent.
                        vec3 pigment=clamp(tint.rgb,vec3(0.0),vec3(1.0));
                        float density=1.0-dot(pigment,
                            vec3(.2126,.7152,.0722));
                        float coating=clamp(.35+.40*density,.35,.75);
                        float mirror_weight=pass*(1.0-fresnel)*
                                            (1.0-coating);
                        color=vec4(mix(glass_ldr,mirror_ldr*pigment,
                                       mirror_weight),1.0);
                        return;
                    }
                }
            }
        }
    }
    color=vec4(glass_ldr,1.0);
}
