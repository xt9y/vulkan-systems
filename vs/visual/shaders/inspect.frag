#version 450
layout(location=0) in vec3 ink;
layout(location=1) in vec2 disc;
layout(location=2) flat in uint kind;
layout(location=3) in float opacity;
layout(location=0) out vec4 color;
void main()
{
    if(kind!=1u && dot(disc,disc)>1.0)discard;
    color=vec4(ink,opacity);
}
