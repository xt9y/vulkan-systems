
#version 450
layout(location=0) out vec4 out_color;
layout(rgba16f,set=0,binding=0) uniform readonly image2D history;
layout(push_constant) uniform Display {vec4 screen;} view;

vec3 fetch_pixel(ivec2 p)
{
    ivec2 size=imageSize(history);
    return imageLoad(history,clamp(p,ivec2(0),size-1)).rgb;
}
void main()
{
    // Bilinear upscale from the reduced tracing resolution.
    vec2 pos=gl_FragCoord.xy/view.screen.xy*vec2(imageSize(history))-0.5;
    ivec2 p=ivec2(floor(pos));
    vec2 t=fract(pos);
    vec3 c=mix(mix(fetch_pixel(p),fetch_pixel(p+ivec2(1,0)),t.x),
               mix(fetch_pixel(p+ivec2(0,1)),
                   fetch_pixel(p+ivec2(1,1)),t.x),t.y);

    // Simple display transform; the HDR history remains linear.
    c=max(c,vec3(0));
    c=c/(vec3(1)+c);
    if(view.screen.z>0.5)c=pow(c,vec3(1.0/2.2));
    out_color=vec4(c,1);
}
