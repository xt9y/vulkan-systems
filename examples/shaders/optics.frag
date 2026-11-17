#version 450
layout(set=0,binding=0) uniform sampler2D indirect;
layout(std430,set=0,binding=1) readonly buffer Caustics {uint energy[];};
layout(location=0) in vec2 uv;
layout(location=0) out vec4 result;
void main() {
    vec3 radiance=max(texture(indirect,uv).rgb,vec3(0));
    ivec2 size=textureSize(indirect,0);
    uvec2 pixel=uvec2(clamp(ivec2(gl_FragCoord.xy),ivec2(0),size-1));
    uint index=(pixel.y*uint(size.x)+pixel.x)*3u;
    vec3 light=vec3(energy[index],energy[index+1],energy[index+2])*.00001;
    vec3 color=(radiance+light)/(vec3(1)+radiance+light);
    result=vec4(color,1);
}
