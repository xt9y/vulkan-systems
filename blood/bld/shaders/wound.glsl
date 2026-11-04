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
    // Two 16-bit fixed-point physical film depths (metres). No 8-bit
    // stamp/opacity mask: smooth coverage emerges from transported volume.
    float mobile=float(packed>>16u)*(0.008/65535.0);
    float retained=float(packed&65535u)*(0.004/65535.0);
    float coverage=1.0-exp(-mobile*4300.0-retained*7000.0);
    float wet=1.0-exp(-mobile*2500.0);
    return vec4(coverage,wet,clamp(mobile/.002,0.0,1.0),
                clamp(retained/.0015,0.0,1.0));
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

 

vec3 bld_film_normal(vec2 uv,int resolution,uint instance_index,
                     vec3 world_position,vec3 world_normal)
{
    // Reconstruct the mobile-film gradient in world metres. The source
    // normal map remains visible underneath thin mobile layers.
    vec2 step_uv=vec2(1.0/float(resolution));
    float hx=(bld_sample(uv+vec2(step_uv.x,0),resolution,instance_index).z-
              bld_sample(uv-vec2(step_uv.x,0),resolution,instance_index).z)
             *.002/(2.0*step_uv.x);
    float hy=(bld_sample(uv+vec2(0,step_uv.y),resolution,instance_index).z-
              bld_sample(uv-vec2(0,step_uv.y),resolution,instance_index).z)
             *.002/(2.0*step_uv.y);
    vec2 dx=dFdx(uv),dy=dFdy(uv);
    float det=dx.x*dy.y-dx.y*dy.x;
    if(abs(det)<1e-8)return world_normal;
    vec3 px=dFdx(world_position),py=dFdy(world_position);
    vec3 du=(px*dy.y-py*dx.y)/det;
    vec3 dv=(py*dx.x-px*dy.x)/det;
    vec3 gradient=du*(hx/max(dot(du,du),1e-8))+
                  dv*(hy/max(dot(dv,dv),1e-8));
    return normalize(world_normal-clamp(length(gradient),0.0,1.5)*
                     normalize(gradient+vec3(1e-8)));
}

void bld_material_film(vec2 uv,int resolution,uint instance_index,
                       vec3 world_position,inout vec3 normal,
                       inout vec3 albedo,inout float roughness,
                       inout float metallic)
{
    normal=bld_film_normal(uv,resolution,instance_index,world_position,normal);
    bld_material(uv,resolution,instance_index,albedo,roughness,metallic);
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
    const uint resolution=48u;
    const uint stride=12u+resolution*resolution;
    for(uint i=0u;i<count;++i)
    {
        uint at=1u+2u*total+stride*(offset+i);
        vec3 center=vec3(uintBitsToFloat(bld_projected[at]),
                         uintBitsToFloat(bld_projected[at+1u]),
                         uintBitsToFloat(bld_projected[at+2u]));
        float width=max(.001,uintBitsToFloat(bld_projected[at+3u]));
        vec3 tangent=normalize(vec3(
            uintBitsToFloat(bld_projected[at+4u]),
            uintBitsToFloat(bld_projected[at+5u]),
            uintBitsToFloat(bld_projected[at+6u])));
        float height=max(.001,uintBitsToFloat(bld_projected[at+7u]));
        vec3 n=normalize(vec3(
            uintBitsToFloat(bld_projected[at+8u]),
            uintBitsToFloat(bld_projected[at+9u]),
            uintBitsToFloat(bld_projected[at+10u])));
        if(dot(n,surface_normal)<.5)continue;
        vec3 delta=world_position-center;
        float signed_distance=dot(delta,n);
        if(abs(signed_distance)>max(.012,width*.08))continue;
        vec3 bitangent=normalize(cross(n,tangent));
        vec2 uv=vec2(dot(delta,tangent)/width+.5,
                     dot(delta,bitangent)/height+.5);
        if(any(lessThan(uv,vec2(0))) || any(greaterThan(uv,vec2(1))))
            continue;
        vec2 p=clamp(uv,vec2(0),vec2(1))*float(resolution-1u);
        uvec2 ij=uvec2(floor(p));
        uvec2 next=min(ij+uvec2(1),uvec2(resolution-1u));
        vec2 weight=fract(p);
        uint base=at+12u;
        uint a=bld_projected[base+ij.y*resolution+ij.x];
        uint b=bld_projected[base+ij.y*resolution+next.x];
        uint c=bld_projected[base+next.y*resolution+ij.x];
        uint d=bld_projected[base+next.y*resolution+next.x];
        vec2 ac=vec2(float(a>>16u),float(a&65535u));
        vec2 bc=vec2(float(b>>16u),float(b&65535u));
        vec2 cc=vec2(float(c>>16u),float(c&65535u));
        vec2 dc=vec2(float(d>>16u),float(d&65535u));
        vec2 layers=mix(mix(ac,bc,weight.x),mix(cc,dc,weight.x),weight.y);
        float mobile=layers.x*(.008/65535.0);
        float retained=layers.y*(.004/65535.0);
        float spot=1.0-exp(-mobile*4300.0-retained*7000.0);
        float damp=1.0-exp(-mobile*2500.0);
        float proximity=1.0-smoothstep(.002,max(.012,width*.08),
                                       abs(signed_distance));
        spot*=proximity;
        coverage=1.0-(1.0-coverage)*(1.0-spot);
        wetness=max(wetness,damp*spot);
    }
    if(coverage<=0.0)return;
    vec3 pigment=mix(vec3(.038,.008,.006),
                     vec3(.125,.005,.011),wetness);
    albedo=mix(albedo,pigment,coverage);
    roughness=mix(roughness,mix(.76,.13,wetness),coverage);
    metallic=mix(metallic,0.0,coverage);
}
