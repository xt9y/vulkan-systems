#version 450
// Exact full-resolution planar mirror: its reflected camera has already
// rasterized the world and PCloud has shaded the same materials in HDR.
// The virtual camera is reflected across a plane (negative determinant),
// so sample horizontally mirrored coordinates to restore handedness.
layout(location=0) in vec2 uv_in;
layout(location=0) out vec4 color;
layout(set=0,binding=0) uniform sampler2D lit_image;
layout(set=0,binding=1) uniform sampler2D opaque_depth;
layout(set=0,binding=2) uniform sampler2D optic_normal;
layout(set=0,binding=5) uniform sampler2D optic_depth;
layout(set=0,binding=19) uniform sampler2D planar_reflection;
void main()
{
    vec2 uv=gl_FragCoord.xy/vec2(textureSize(lit_image,0));
    if(texture(optic_normal,uv).a<.75)discard;
    if(texture(optic_depth,uv).r>texture(opaque_depth,uv).r+.0003)
        discard;
    color=vec4(textureLod(planar_reflection,
              vec2(1.0-uv.x,uv.y),0.0).rgb,1.0);
}
