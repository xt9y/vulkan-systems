#version 450
#extension GL_GOOGLE_include_directive : require
// Full-resolution planar mirror/window: raster-reflected world with
// non-refracting, same-pixel transmission through single-panel glass.
layout(location=0) in vec2 uv_in;
layout(location=0) out vec4 color;
layout(set=0,binding=0) uniform sampler2D lit_image;
layout(set=0,binding=1) uniform sampler2D opaque_depth;
layout(set=0,binding=2) uniform sampler2D optic_normal;
layout(set=0,binding=3) uniform sampler2D optic_tint;
layout(set=0,binding=5) uniform sampler2D optic_depth;
layout(set=0,binding=17) uniform sampler2D planar_reflection;
layout(set=0,binding=22) uniform sampler2D reflected_normal;
layout(set=0,binding=23) uniform sampler2D reflected_tint;
layout(set=0,binding=24) uniform sampler2D reflected_material;
layout(set=0,binding=25) uniform sampler2D reflected_depth;
layout(set=0,binding=26) uniform sampler2D reflected_opaque_depth;
#include "glass_layer.glsl"
layout(push_constant) uniform View {
    mat4 vp;
    vec4 eye, right, up, forward;
} camera;
void main()
{
    vec2 uv=gl_FragCoord.xy/vec2(textureSize(lit_image,0));
    // The glass pass already resolved the transmissive pixels.
    vec4 mask=texture(optic_normal,uv);
    if(mask.a<.75)discard;
    if(texture(optic_depth,uv).r>
       texture(opaque_depth,uv).r+.0003)discard;
    vec4 tint=texture(optic_tint,uv);
    vec2 reflected_uv=vec2(1.0-uv.x,uv.y);
    vec3 reflected=textureLod(planar_reflection,reflected_uv,0.0).rgb;
    // The reflected camera also rasterizes optical surfaces. Composite
    // their nearest glass layer over the reflected scene before applying
    // the mirror's own authored tint.
    vec4 mask_glass=texture(reflected_normal,reflected_uv);
    float depth_glass=texture(reflected_depth,reflected_uv).r;
    float depth_opaque=texture(reflected_opaque_depth,reflected_uv).r;
    if(mask_glass.a>.25 && mask_glass.a<.75 &&
       depth_glass<depth_opaque+.0003) {
        vec4 glass_tint=texture(reflected_tint,reflected_uv);
        vec4 glass_material=texture(reflected_material,reflected_uv);
        vec3 plane_normal=normalize(camera.vp[0].xyz);
        vec3 view_forward=reflect(camera.forward.xyz,plane_normal);
        vec3 view_up=reflect(camera.up.xyz,plane_normal);
        vec3 view_right=normalize(cross(view_forward,view_up));
        view_up=normalize(cross(view_right,view_forward));
        vec2 screen=reflected_uv*2.0-1.0;
        vec3 incident=normalize(view_forward+
            view_right*screen.x*camera.right.w-
            view_up*screen.y*camera.up.w);
        optics_glass_profile surface=optics_glass_shape(
            mask_glass,incident,view_right,view_up,
            vec2(camera.right.w,camera.up.w),
            clamp(1.0+glass_material.g,1.001,2.0),
            glass_material.b);
        vec2 border=.5/vec2(textureSize(planar_reflection,0));
        vec3 through=reflected,sheen=reflected*.7;
        if(glass_material.b>.5) {
            through=texture(planar_reflection,
                clamp(reflected_uv+surface.shift,border,1.0-border)).rgb;
            sheen=texture(planar_reflection,
                clamp(reflected_uv-surface.shift*.4,border,1.0-border)).rgb*.7;
        }
        reflected=optics_glass_shaped(through,sheen,glass_tint.rgb,
            glass_tint.a,glass_material.b,surface.fresnel);
    }
    color=vec4(reflected*tint.rgb,1.0);
}
