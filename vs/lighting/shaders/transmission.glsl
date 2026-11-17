#ifndef VS_RGB_TRANSMISSION_GLSL
#define VS_RGB_TRANSMISSION_GLSL
#ifndef VS_TRANSMISSION_BINDING
#define VS_TRANSMISSION_BINDING 11
#endif
// Shared PCloud/renderer light-space transmission cache.
// 64 active flags + up to four RGB8/depth8 crossings per octahedral texel.
// A zero active flag preserves the old lighting path, with no cache reads.
const uint VS_TRANSMISSION_RESOLUTION=128u;
const uint VS_TRANSMISSION_LAYERS=8u;
layout(std430,set=0,binding=VS_TRANSMISSION_BINDING)
readonly buffer VsTransmission {
    uint map_index[64];
    uint samples[];
} vs_transmission;

vec2 vs_oct_encode(vec3 normal)
{
    normal/=abs(normal.x)+abs(normal.y)+abs(normal.z);
    vec2 encoded=normal.xy;
    if(normal.z<0.0)
        encoded=(1.0-abs(encoded.yx))*
            mix(vec2(-1.0),vec2(1.0),greaterThanEqual(encoded,vec2(0.0)));
    return encoded*.5+.5;
}
vec3 vs_transmission_corner(uint light,ivec2 pixel,float length_to_receiver,
                            float radius)
{
    pixel=clamp(pixel,ivec2(0),ivec2(int(VS_TRANSMISSION_RESOLUTION)-1));
    const uint stride=VS_TRANSMISSION_RESOLUTION*VS_TRANSMISSION_RESOLUTION*
                      VS_TRANSMISSION_LAYERS;
    uint first=(vs_transmission.map_index[light]-1u)*stride+
        (uint(pixel.y)*VS_TRANSMISSION_RESOLUTION+uint(pixel.x))*
        VS_TRANSMISSION_LAYERS;
    vec3 color=vec3(1.0);
    for(uint layer=0u;layer<VS_TRANSMISSION_LAYERS;++layer) {
        uint value=vs_transmission.samples[first+layer];
        uint depth=value>>24u;
        if(depth==255u)break;
        if(length_to_receiver+0.5*radius/254.0<
           float(depth)*radius/254.0)continue;
        vec3 rgb=vec3(float(value&255u),
                      float((value>>8u)&255u),
                      float((value>>16u)&255u))*(1.0/255.0);
        color*=rgb;
    }
    return color;
}
// Direction is from light to receiver, not a screen-space coordinate.
// Bilinear filter produces stable, soft colored penumbras without additional
// ray queries in the per-pixel light loop.
vec3 vs_light_transmission(uint light,vec3 from_light,float radius)
{
    if(light>=64u || vs_transmission.map_index[light]==0u ||
       !(radius>0.0))return vec3(1.0);
    float distance=length(from_light);
    if(distance<1e-5)return vec3(1.0);
    vec2 uv=vs_oct_encode(from_light/distance)*
        float(VS_TRANSMISSION_RESOLUTION)-vec2(0.5);
    ivec2 base=ivec2(floor(uv));
    vec2 fraction=fract(uv);
    // Four depth-aware taps, reused for both filtering and silhouette
    // classification; no additional SSBO fetches in the edge path.
    vec3 c00=vs_transmission_corner(light,base,distance,radius);
    vec3 c10=vs_transmission_corner(light,base+ivec2(1,0),distance,radius);
    vec3 c01=vs_transmission_corner(light,base+ivec2(0,1),distance,radius);
    vec3 c11=vs_transmission_corner(light,base+ivec2(1,1),distance,radius);
    vec3 filtered=mix(mix(c00,c10,fraction.x),
                      mix(c01,c11,fraction.x),fraction.y);
    vec3 nearest=fraction.x<.5?
        (fraction.y<.5?c00:c01):
        (fraction.y<.5?c10:c11);
    float edge=max(max(length(c00-c10),length(c00-c01)),
                   max(length(c11-c10),length(c11-c01)));
    return mix(filtered,nearest,smoothstep(0.08,0.3,edge));
}
#endif
