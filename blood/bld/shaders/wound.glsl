// Surface wetness stored in the mesh UV atlas, attached through animations.
// One 32-bit word per texel: coverage, wetness, thickness, porosity.
layout(std430,set=0,binding=14) readonly buffer BloodSurfaceAtlas
{
    uint bld_pixels[];
};

vec4 bld_texel(ivec2 p,int resolution,uint instance_index)
{
    p=clamp(p,ivec2(0),ivec2(resolution-1));
    uint total=bld_pixels[0];
    if(instance_index>=total)return vec4(0.0);
    uint slot=bld_pixels[1u+instance_index];
    if(slot==0xffffffffu)return vec4(0.0);
    uint index=1u+total+slot*uint(resolution*resolution)+
               uint(p.y*resolution+p.x);
    uint packed=bld_pixels[index];
    return vec4(float((packed>>24u)&255u),
                float((packed>>16u)&255u),
                float((packed>>8u)&255u),
                float(packed&255u))/255.0;
}

vec4 bld_sample(vec2 uv,int resolution,uint instance_index)
{
    vec2 p=clamp(uv,vec2(0),vec2(1))*float(resolution-1);
    ivec2 origin=ivec2(floor(p));
    vec2 t=fract(p);
    vec4 a=mix(bld_texel(origin,resolution,instance_index),
               bld_texel(origin+ivec2(1,0),resolution,instance_index),t.x);
    vec4 b=mix(bld_texel(origin+ivec2(0,1),resolution,instance_index),
               bld_texel(origin+ivec2(1,1),resolution,instance_index),t.x);
    return mix(a,b,t.y);
}

void bld_material(vec2 uv,int resolution,uint instance_index,
                  inout vec3 albedo,inout float roughness,
                  inout float metallic)
{
    vec4 stain=bld_sample(uv,resolution,instance_index);
    float coverage=stain.x;
    float wet=stain.y;
    float depth=stain.z;
    float absorbency=stain.w;
    vec3 fresh=vec3(.125,.005,.011);
    vec3 dry=vec3(.038,.008,.006);
    vec3 pigment=mix(dry,fresh,wet)*mix(.91,.63,depth);
    albedo=mix(albedo,pigment,coverage);
    float wet_roughness=mix(.76,.13,wet*(1.0-absorbency*.86));
    roughness=mix(roughness,wet_roughness,coverage);
    metallic=mix(metallic,0.0,coverage);
}
