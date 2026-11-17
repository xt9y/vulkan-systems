#ifndef PC_LIGHTING_SHARED_GLSL
#define PC_LIGHTING_SHARED_GLSL
// Shared by PCloud primary lighting and VulkanOptics secondary hits.
// Keep the PBR BRDF, probe interpolation and tone curve identical.
const float shared_pi=3.14159265359;
vec3 shared_fresnel(float cosine,vec3 f0)
{
    float m=1.0-clamp(cosine,0.0,1.0);
    float m2=m*m;
    return f0+(1.0-f0)*(m2*m2*m);
}
vec3 shared_brdf(vec3 albedo,float metallic,float roughness,
                 vec3 N,vec3 V,vec3 L)
{
    float nl=max(dot(N,L),0.0);
    if(nl<=0.0)return vec3(0.0);
    vec3 H=normalize(V+L);
    float nv=max(dot(N,V),1e-4);
    float nh=max(dot(N,H),0.0);
    float vh=max(dot(V,H),0.0);
    float a=roughness*roughness,a2=a*a;
    float d=nh*nh*(a2-1.0)+1.0;
    float D=a2/max(shared_pi*d*d,1e-6);
    float k=(roughness+1.0)*(roughness+1.0)*0.125;
    float G=(nv/(nv*(1.0-k)+k))*(nl/(nl*(1.0-k)+k));
    vec3 F=shared_fresnel(vh,mix(vec3(0.04),albedo,metallic));
    return ((1.0-F)*(1.0-metallic)*albedo/shared_pi+
            D*G*F/max(4.0*nv*nl,1e-4))*nl;
}
// Exact Rogue eight-probe blend and backface rejection.
vec3 shared_indirect_lighting(sampler3D grid,vec3 origin,float spacing,
                              vec3 world,vec3 normal)
{
    if(spacing<=0.0)return vec3(0.0);
    vec3 coords=(world-origin)/spacing;
    ivec3 base=ivec3(floor(coords));
    vec3 blend=fract(coords);
    ivec3 size=textureSize(grid,0);
    vec3 sum=vec3(0.0);
    float weight_sum=0.0;
    for(int z=0;z<2;++z)
    for(int y=0;y<2;++y)
    for(int x=0;x<2;++x)
    {
        ivec3 offset=ivec3(x,y,z),probe=base+offset;
        if(any(lessThan(probe,ivec3(0))) ||
           any(greaterThanEqual(probe,size)))continue;
        vec3 to_probe=origin+vec3(probe)*spacing-world;
        if(dot(normal,to_probe)<=0.0)continue;
        vec3 factors=mix(vec3(1.0)-blend,blend,vec3(offset));
        float weight=factors.x*factors.y*factors.z;
        sum+=weight*texelFetch(grid,probe,0).rgb;
        weight_sum+=weight;
    }
    return weight_sum>1e-5?max(sum/weight_sum,vec3(0.0)):vec3(0.0);
}
vec3 shared_tonemap(vec3 radiance,float exposure)
{
    vec3 exposed=max(radiance*exposure,vec3(0.0));
    return exposed/(vec3(1.0)+exposed);
}
#endif
