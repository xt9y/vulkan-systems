#version 450
layout(location=0) in vec2 uv;
layout(set=0,binding=0) uniform sampler2D source_image;
layout(location=0) out vec4 color;
void main()
{
    vec3 hdr=max(texture(source_image,uv).rgb,vec3(0));
    // Source is linear HDR, swapchain attachment is sRGB.
    color=vec4(hdr/(vec3(1)+hdr),1);
}
