#pragma once

// These GLSL sources belong to VulkanLumen, not the test application.
// Stage 1: rasterized Cornell geometry and a fullscreen HDR composite.
namespace lumen::shaders {

inline constexpr const char* geometry_vert=R"glsl(#version 450
layout(location=0) in vec3 position;
layout(location=1) in vec3 color;
layout(location=0) out vec3 surface_color;

layout(push_constant) uniform Camera {
    vec4 eye;
    vec4 right;
    vec4 up;
    vec4 projection;  // tan(fov/2), aspect, near, far
} camera;

void main()
{
    vec3 relative=position-camera.eye.xyz;
    vec3 forward=cross(camera.up.xyz,camera.right.xyz);
    float view_x=dot(relative,camera.right.xyz);
    float view_y=dot(relative,camera.up.xyz);
    float view_z=dot(relative,forward);
    float n=camera.projection.z;
    float f=camera.projection.w;
    float z=view_z*f/(f-n)-n*f/(f-n);
    // Vulkan viewport maps negative clip Y to the top of the image.
    gl_Position=vec4(view_x/(camera.projection.x*camera.projection.y),
                    -view_y/camera.projection.x,z,view_z);
    surface_color=color;
}
)glsl";

inline constexpr const char* geometry_frag=R"glsl(#version 450
layout(location=0) in vec3 surface_color;
layout(location=0) out vec4 hdr_color;
void main()
{
    // Unlit material preview; actual direct lighting is Stage 2.
    hdr_color=vec4(surface_color,1.0);
}
)glsl";

inline constexpr const char* composite_vert=R"glsl(#version 450
layout(location=0) out vec2 uv;
void main()
{
    const vec2 triangle[3]=vec2[3](
        vec2(-1.0,-1.0),vec2(3.0,-1.0),vec2(-1.0,3.0));
    vec2 p=triangle[gl_VertexIndex];
    uv=p*0.5+0.5;
    gl_Position=vec4(p,0.0,1.0);
}
)glsl";

inline constexpr const char* composite_frag=R"glsl(#version 450
layout(set=0,binding=0) uniform sampler2D color_image;
layout(push_constant) uniform Display {
    uint manual_gamma;
} display;
layout(location=0) in vec2 uv;
layout(location=0) out vec4 color_out;
void main()
{
    vec3 hdr=max(texture(color_image,uv).rgb,vec3(0.0));
    vec3 mapped=hdr/(vec3(1.0)+hdr);
    if(display.manual_gamma!=0u)
        mapped=pow(mapped,vec3(1.0/2.2));
    color_out=vec4(mapped,1.0);
}
)glsl";

} // namespace lumen::shaders
