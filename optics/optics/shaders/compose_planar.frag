#version 450
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
        reflected=optics_glass_layer(reflected,glass_tint.rgb,
                                    glass_tint.a,glass_material.b);
    }
    color=vec4(reflected*tint.rgb,1.0);
}
