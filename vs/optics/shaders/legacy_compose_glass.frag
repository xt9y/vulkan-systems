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
#include "legacy_glass_layer.glsl"

// Refracted HDR taps may only contain geometry *behind* the glass surface.
// Reject foreground world objects and the FPS weapon (whose custom depth
// range is [0,.099]); displaced samples otherwise create translucent copies.
// Retain hardware bilinear filtering when all four taps pass this depth test.
vec3 glass_world_hdr(vec2 requested,ivec2 receiver,float surface_depth)
{
    ivec2 dimensions=textureSize(scene_hdr,0);
    vec2 p=clamp(requested*vec2(dimensions)-vec2(.5),vec2(0),
                 vec2(dimensions)-vec2(1));
    ivec2 base=ivec2(floor(p));
    ivec2 p00=base;
    ivec2 p10=min(base+ivec2(1,0),dimensions-ivec2(1));
    ivec2 p01=min(base+ivec2(0,1),dimensions-ivec2(1));
    ivec2 p11=min(base+ivec2(1,1),dimensions-ivec2(1));
    bvec4 foreground=bvec4(
        texelFetch(opaque_depth,p00,0).r<surface_depth-.0003,
        texelFetch(opaque_depth,p10,0).r<surface_depth-.0003,
        texelFetch(opaque_depth,p01,0).r<surface_depth-.0003,
        texelFetch(opaque_depth,p11,0).r<surface_depth-.0003);
    if(!any(foreground))return texture(scene_hdr,requested).rgb;
    vec3 fallback=texelFetch(scene_hdr,receiver,0).rgb;
    vec3 a=foreground.x?fallback:texelFetch(scene_hdr,p00,0).rgb;
    vec3 b=foreground.y?fallback:texelFetch(scene_hdr,p10,0).rgb;
    vec3 c=foreground.z?fallback:texelFetch(scene_hdr,p01,0).rgb;
    vec3 d=foreground.w?fallback:texelFetch(scene_hdr,p11,0).rgb;
    vec2 weight=fract(p);
    return mix(mix(a,b,weight.x),mix(c,d,weight.x),weight.y);
}
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
    vec2 screen=uv*2.0-1.0;
    vec3 incoming=normalize(camera.forward.xyz+
        camera.right.xyz*(screen.x*camera.right.w)-
        camera.up.xyz*(screen.y*camera.up.w));

    float ior=clamp(1.0+material.g,1.001,2.0);
    bool solid=material.b>.5;
    // Shared Fresnel/refraction shape is used for primary, nested and
    // reflected glass. Keep displacement bounded and preserve perspective.
    optics_glass_profile profile=optics_glass_shape(
        mask,incoming,camera.right.xyz,camera.up.xyz,
        vec2(camera.right.w,camera.up.w),ior,material.b);
    float fresnel=profile.fresnel;
    vec2 shift=profile.shift;
    vec2 border=.5/size;
    vec2 transmitted_uv=clamp(uv+shift,border,1.0-border);
    // Both ordinary geometry and planar mirrors must undergo the same
    // glass shading. Convert the normal HDR scene into the same display
    // space as the already-tonemapped planar reflection, then substitute
    // only the background source if the refracted view hits the mirror.
    // A flat pane has zero screen displacement. Read its same-pixel HDR
    // directly instead of four depth taps plus the foreground-resolve path.
    vec3 transmitted_hdr=solid?
        glass_world_hdr(transmitted_uv,pixel,depth):
        texelFetch(scene_hdr,pixel,0).rgb;
    if(solid && material.g>.01) {
        // Bounded RGB dispersion: red and blue use slightly different IORs.
        // Only two extra HDR reads on glass pixels; no per-pixel ray tracing.
        const float cauchy_delta=.006;
        vec2 red_shift=optics_glass_shape(mask,incoming,camera.right.xyz,
            camera.up.xyz,vec2(camera.right.w,camera.up.w),
            max(1.001,ior-cauchy_delta),material.b).shift;
        vec2 blue_shift=optics_glass_shape(mask,incoming,camera.right.xyz,
            camera.up.xyz,vec2(camera.right.w,camera.up.w),
            min(2.0,ior+cauchy_delta),material.b).shift;
        transmitted_hdr.r=glass_world_hdr(
            clamp(uv+red_shift,border,1.0-border),pixel,depth).r;
        transmitted_hdr.b=glass_world_hdr(
            clamp(uv+blue_shift,border,1.0-border),pixel,depth).b;
    }
    vec3 background_ldr=shared_tonemap(transmitted_hdr,camera.forward.w);
    float pass=clamp(tint.a,0.0,1.0);
    float mirror_surface_z=1e30;
    vec2 mirror_image_uv=vec2(0.0);
    vec3 reflection_ldr=shared_tonemap(
        (solid?glass_world_hdr(
            clamp(uv-shift*.4,border,1.0-border),pixel,depth):
            transmitted_hdr)*.7,camera.forward.w);

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
                            vec3 mirror_n=normalize(camera.vp[0].xyz);
                            vec3 virtual_forward=reflect(camera.forward.xyz,mirror_n);
                            vec3 virtual_up=reflect(camera.up.xyz,mirror_n);
                            vec3 virtual_right=normalize(cross(virtual_forward,virtual_up));
                            virtual_up=normalize(cross(virtual_right,virtual_forward));
                            vec2 screen=reflected_uv*2.0-1.0;
                            vec3 incident=normalize(virtual_forward+
                                virtual_right*screen.x*camera.right.w-
                                virtual_up*screen.y*camera.up.w);
                            optics_glass_profile surface=optics_glass_shape(
                                back_mask,incident,virtual_right,virtual_up,
                                vec2(camera.right.w,camera.up.w),
                                clamp(1.0+back_material.g,1.001,2.0),
                                back_material.b);
                            vec2 border_ref=.5/vec2(textureSize(planar_reflection,0));
                            vec3 mirror_tint=clamp(camera.vp[3].rgb,
                                vec3(0.0),vec3(1.0));
                            vec3 through=mirror_ldr,sheen=mirror_ldr*.7;
                            if(back_material.b>.5) {
                                through=texture(planar_reflection,
                                    clamp(reflected_uv+surface.shift,
                                          border_ref,1.0-border_ref)).rgb*mirror_tint;
                                sheen=texture(planar_reflection,
                                    clamp(reflected_uv-surface.shift*.4,
                                          border_ref,1.0-border_ref)).rgb*
                                          mirror_tint*.7;
                            }
                            mirror_ldr=optics_glass_shaped(through,sheen,
                                back_tint.rgb,back_tint.a,back_material.b,
                                surface.fresnel);
                        }
                        background_ldr=mirror_ldr;
                        mirror_surface_z=mirror_z;
                        mirror_image_uv=reflected_uv;
                    }
                }
            }
        }
    }
    // Sample the second FRONT-facing glass layer behind this one. Its
    // own depth test guarantees the layer belongs behind the first
    // optical surface; opaque geometry still wins if nearer.
    // When projected glass panes do not overlap, their second layer is
    // empty and needs no texture/depth reads.
    if(camera.eye.w<2.5) {
    vec4 second_mask=texture(second_normal,transmitted_uv);
    // Most primary glass fragments have no second layer. Only fetch
    // secondary and opaque depth for pixels that actually contain glass.
    if(second_mask.a>.25 && second_mask.a<.75) {
    float second_z=texture(second_depth,transmitted_uv).r;
    float opaque_z=texture(opaque_depth,transmitted_uv).r;
    if(second_z>depth+.0001 && second_z<opaque_z+.0003) {
    float second_ndc=clamp((second_z-.1)/.9,0.0,1.0);
    float second_linear=100.0/
        max(1000.0-999.9*second_ndc,1e-5);
    if(second_linear+.02<mirror_surface_z) {
        vec4 behind_tint=texture(second_tint,transmitted_uv);
        vec4 behind_material=texture(second_material,transmitted_uv);
        vec2 second_screen=transmitted_uv*2.0-1.0;
        vec3 second_ray=normalize(camera.forward.xyz+
            camera.right.xyz*second_screen.x*camera.right.w-
            camera.up.xyz*second_screen.y*camera.up.w);
        optics_glass_profile behind=optics_glass_shape(
            second_mask,second_ray,camera.right.xyz,camera.up.xyz,
            vec2(camera.right.w,camera.up.w),
            clamp(1.0+behind_material.g,1.001,2.0),
            behind_material.b);
        vec3 through=background_ldr,sheen=background_ldr*.7;
        if(behind_material.b>.5 && mirror_surface_z<1e29) {
            // Mirror coordinates are horizontally flipped relative to
            // the primary view; use the already-rendered mirror image.
            vec2 mirror_shift=vec2(-behind.shift.x,behind.shift.y);
            vec3 mirror_tint=clamp(camera.vp[3].rgb,vec3(0),vec3(1));
            through=texture(planar_reflection,clamp(mirror_image_uv+
                mirror_shift,border,1.0-border)).rgb*mirror_tint;
            sheen=texture(planar_reflection,clamp(mirror_image_uv-
                mirror_shift*.4,border,1.0-border)).rgb*mirror_tint*.7;
        } else if(behind_material.b>.5) {
            through=shared_tonemap(glass_world_hdr(
                clamp(transmitted_uv+behind.shift,border,1.0-border),pixel,second_z),
                camera.forward.w);
            sheen=shared_tonemap(glass_world_hdr(
                clamp(transmitted_uv-behind.shift*.4,border,1.0-border),pixel,second_z)*
                .7,camera.forward.w);
        }
        background_ldr=optics_glass_shaped(
            mix(background_ldr,through,.85),sheen,
            behind_tint.rgb,behind_tint.a,behind_material.b,
            behind.fresnel);
    }
    }
    }
    }
    // Transmitted scene radiance, absorption tint and Fresnel define
    // glass appearance without tracing separate surface-light rays.
    vec3 glass=optics_glass_layer(background_ldr,tint.rgb,pass,
                                 solid?1.0:0.0);
    color=vec4(clamp(mix(glass,reflection_ldr,fresnel),
                     vec3(0.0),vec3(1.0)),1.0);
}
