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
layout(set=0,binding=18) uniform sampler2D planar_reflection_hdr;
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
    vec3 travel_dir=incoming;
    if(solid) {
        vec3 transmitted_ray=refract(incoming,normal,1.0/ior);
        if(dot(transmitted_ray,transmitted_ray)>1e-6)
            travel_dir=normalize(transmitted_ray);
        vec3 bend=travel_dir-incoming;
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

    // The opaque HDR image contains no mirrors: those are composited
    // after lighting. If this refracted view hits the planar mirror,
    // sample the reflected camera in *its own projected coordinates*.
    // Read linear HDR and apply the common tone mapper only once.
    // camera.vp carries plane, center, and Y/Z axes/half extents here;
    // no additional push constants or per-glass-pixel BVH queries.
    if(camera.vp[1].w>.5) {
        vec4 plane=camera.vp[0];
        vec3 normal_m=plane.xyz;
        float ndc=clamp((depth-.1)/.9,0.0,1.0);
        float linear=.1*1000.0/max(1000.0-ndc*999.9,1e-5);
        vec3 surface_pos=camera.eye.xyz+camera.forward.xyz*linear+
            camera.right.xyz*screen.x*linear*camera.right.w-
            camera.up.xyz*screen.y*linear*camera.up.w;
        float facing=dot(normal_m,travel_dir);
        if(abs(facing)>1e-5) {
            float distance=-(dot(normal_m,surface_pos)+plane.w)/facing;
            if(distance>.01) {
                vec3 hit=surface_pos+travel_dir*distance;
                vec3 rel=hit-camera.vp[1].xyz;
                if(abs(dot(rel,camera.vp[2].xyz))<=camera.vp[2].w &&
                   abs(dot(rel,camera.vp[3].xyz))<=camera.vp[3].w) {
                    // Project the mirror hit into the primary camera
                    // only to test whether opaque geometry blocks it.
                    vec3 to_hit=hit-camera.eye.xyz;
                    float primary_z=dot(to_hit,camera.forward.xyz);
                    if(primary_z>.1) {
                        vec2 primary_uv=vec2(
                            .5+.5*dot(to_hit,camera.right.xyz)/
                                (primary_z*camera.right.w),
                            .5-.5*dot(to_hit,camera.up.xyz)/
                                (primary_z*camera.up.w));
                        if(all(greaterThanEqual(primary_uv,vec2(0.0))) &&
                           all(lessThanEqual(primary_uv,vec2(1.0)))) {
                            float opaque=texture(opaque_depth,primary_uv).r;
                            float opaque_ndc=clamp((opaque-.1)/.9,0.0,1.0);
                            float opaque_z=100.0/
                                max(1000.0-opaque_ndc*999.9,1e-5);
                            if(opaque_z+.02>=primary_z) {
                                // Reconstruct the same reflected view
                                // orientation as scene_renderer's look_at().
                                float eye_distance=dot(normal_m,camera.eye.xyz)+plane.w;
                                vec3 reflected_eye=camera.eye.xyz-
                                    2.0*eye_distance*normal_m;
                                vec3 reflected_forward=reflect(
                                    camera.forward.xyz,normal_m);
                                vec3 reflected_up=reflect(camera.up.xyz,normal_m);
                                vec3 reflected_right=normalize(cross(
                                    reflected_forward,reflected_up));
                                reflected_up=normalize(cross(
                                    reflected_right,reflected_forward));
                                vec3 in_reflection=hit-reflected_eye;
                                float reflected_z=dot(
                                    in_reflection,reflected_forward);
                                if(reflected_z>.1) {
                                    vec2 reflection_uv=vec2(
                                        .5+.5*dot(in_reflection,reflected_right)/
                                            (reflected_z*camera.right.w),
                                        .5-.5*dot(in_reflection,reflected_up)/
                                            (reflected_z*camera.up.w));
                                    if(all(greaterThanEqual(reflection_uv,vec2(0.0))) &&
                                       all(lessThanEqual(reflection_uv,vec2(1.0)))) {
                                        vec3 mirror_hdr=texture(
                                            planar_reflection_hdr,reflection_uv).rgb;
                                        radiance=mix(radiance,
                                            mirror_hdr*absorption,
                                            pass*(1.0-fresnel));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    color=vec4(shared_tonemap(radiance,camera.forward.w),1.0);
}
