#version 450
layout(set=0,binding=0) uniform sampler2D radiance;
layout(location=0) in vec2 uv;
layout(location=0) out vec4 color;
void main() {
    vec3 hdr=max(texture(radiance,uv).rgb,vec3(0));
    vec3 mapped=hdr/(hdr+vec3(1));
    color=vec4(mapped,1);
}
