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
    vec4 mask=texture(optic_normal,uv);
    if(mask.a<.25)discard;
    if(texture(optic_depth,uv).r>texture(opaque_depth,uv).r+.0003)
        discard;
    vec4 tint=texture(optic_tint,uv);
    vec3 reflection=textureLod(planar_reflection,
        vec2(1.0-uv.x,uv.y),0.0).rgb;
    if(mask.a>=.75) {
        color=vec4(reflection*tint.rgb,1.0);
        return;
    }
    // Planar glass mixes two tonemapped raster images, not BVH paths.
    vec2 screen=uv*2.0-1.0;
    vec3 incoming=normalize(camera.forward.xyz+
        camera.right.xyz*(screen.x*camera.right.w)-
        camera.up.xyz*(screen.y*camera.up.w));
    vec3 N=normalize(mask.rgb*2.0-1.0);
    if(dot(N,incoming)>0.0)N=-N;
    float cosine=clamp(dot(-incoming,N),0.0,1.0);
    const float f0=.04258; // IOR = 1.52
    float fresnel=f0+(1.0-f0)*pow(1.0-cosine,5.0);
    float strength=clamp(tint.a,0.0,1.0);
    vec3 transmitted=textureLod(lit_image,uv,0.0).rgb*
                     mix(vec3(1.0),tint.rgb,.20);
    vec3 glass=mix(tint.rgb,transmitted,strength);
    color=vec4(mix(glass,reflection,fresnel),1.0);
}
