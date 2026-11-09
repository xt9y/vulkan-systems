#version 450
layout(location=0) out vec4 out_color;
layout(rgba16f,set=0,binding=0) uniform readonly image2D display_image;
layout(rgba16f,set=0,binding=1) uniform readonly image2D hit_position;
layout(push_constant) uniform Display {vec4 screen;} view;

ivec2 safe_pixel(ivec2 p)
{
    return clamp(p,ivec2(0),imageSize(display_image)-1);
}
vec3 fetch_pixel(ivec2 p)
{
    return imageLoad(display_image,safe_pixel(p)).rgb;
}
void main()
{
    vec2 pos=gl_FragCoord.xy/view.screen.xy*
             vec2(imageSize(display_image))-.5;
    ivec2 base=ivec2(floor(pos));
    ivec2 nearest=safe_pixel(ivec2(floor(pos+.5)));
    vec2 t=fract(pos);
    float surface=imageLoad(hit_position,nearest).w;
    vec3 sum=vec3(0);
    float total=0.0;
    for(int y=0;y<2;++y)
    for(int x=0;x<2;++x)
    {
        ivec2 q=safe_pixel(base+ivec2(x,y));
        // Do not bleed an emitter or another Cornell surface through
        // a depth-discontinuous edge while scaling to the swapchain.
        if(view.screen.w>.5 &&
           imageLoad(hit_position,q).w!=surface)continue;
        float w=(x==0?1.0-t.x:t.x)*(y==0?1.0-t.y:t.y);
        sum+=fetch_pixel(q)*w;
        total+=w;
    }
    vec3 color=total>1e-5?sum/total:fetch_pixel(nearest);
    color=max(color,vec3(0));
    color=color/(vec3(1)+color);
    if(view.screen.z>.5)color=pow(color,vec3(1.0/2.2));
    out_color=vec4(color,1);
}
