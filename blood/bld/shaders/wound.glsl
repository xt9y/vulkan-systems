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
    // Match repeat-wrapped glTF UVs, including meshes with tiled (>1) coordinates.
    vec2 p=fract(uv)*float(resolution-1);
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

 
// Rigid surfaces use metre-sized world-space footprints, not a UV atlas
// stretched across an entire floor. One word is the instance count;
// each instance has offset/count, followed by packed center/radius/normal/wetness.
layout(std430,set=0,binding=15) readonly buffer BloodProjectedStains
{
    uint bld_projected[];
};

void bld_projected_material(vec3 world_position,vec3 world_normal,
                            uint instance_index,
                            inout vec3 albedo,inout float roughness,
                            inout float metallic)
{
    uint total=bld_projected[0];
    if(instance_index>=total)return;
    uint offset=bld_projected[1u+instance_index];
    uint count=bld_projected[1u+total+instance_index];
    if(count==0u)return;
    vec3 surface_normal=normalize(world_normal);
    float coverage=0.0;
    float wetness=0.0;
    for(uint i=0u;i<count;++i)
    {
        uint at=1u+2u*total+8u*(offset+i);
        vec3 center=vec3(uintBitsToFloat(bld_projected[at]),
                         uintBitsToFloat(bld_projected[at+1u]),
                         uintBitsToFloat(bld_projected[at+2u]));
        float radius=max(.001,uintBitsToFloat(bld_projected[at+3u]));
        vec3 n=normalize(vec3(
            uintBitsToFloat(bld_projected[at+4u]),
            uintBitsToFloat(bld_projected[at+5u]),
            uintBitsToFloat(bld_projected[at+6u])));
        // Reject geometry behind the hit plane and facing a different way.
        float facing=dot(n,surface_normal);
        if(facing<.55)continue;
        vec3 delta=world_position-center;
        float depth=abs(dot(delta,n));
        if(depth>max(.012,radius*.15))continue;
        vec3 in_plane=delta-n*dot(delta,n);
        float angle=atan(in_plane.y,in_plane.x);
        float irregular=1.0+.075*sin(angle*7.0+float(i)*1.7)
                           +.035*sin(angle*13.0+float(i)*.9);
        float r=length(in_plane)/max(.001,radius*irregular);
        float spot=1.0-smoothstep(.68,1.0,r);
        spot*=1.0-smoothstep(0.0,max(.012,radius*.15),depth);
        coverage=1.0-(1.0-coverage)*(1.0-spot);
        wetness=max(wetness,spot*
            clamp(uintBitsToFloat(bld_projected[at+7u]),0.0,1.0));
    }
    if(coverage<=0.0)return;
    vec3 pigment=mix(vec3(.038,.008,.006),
                     vec3(.125,.005,.011),wetness);
    albedo=mix(albedo,pigment,coverage);
    roughness=mix(roughness,mix(.76,.16,wetness),coverage);
    metallic=mix(metallic,0.0,coverage);
}
