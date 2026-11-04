#version 450
#extension GL_GOOGLE_include_directive : require

layout(location=0) in vec2 uv;
layout(location=1) flat in uint instance_index;
layout(location=0) out vec4 color;

#include "wound.glsl"

void main()
{
    vec3 pigment=vec3(.5);
    float roughness=.65;
    float metallic=.0;
    bld_material(uv,256,instance_index,pigment,roughness,metallic);
    color=vec4(pigment,1.);
}
