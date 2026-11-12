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
    color=vec4(textureLod(planar_reflection,
        vec2(1.0-uv.x,uv.y),0.0).rgb*tint.rgb,1.0);
}
