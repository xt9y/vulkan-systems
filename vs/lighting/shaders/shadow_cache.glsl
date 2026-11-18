#ifndef VS_STATIC_SHADOW_CACHE_GLSL
#define VS_STATIC_SHADOW_CACHE_GLSL
// Compact two-key world-space static visibility cache. Only the static BVH
// is cached; animated geometry always receives a fresh shadow traversal.
// A cache miss is an exact fallback, not a guessed visibility value.
layout(std430,set=0,binding=18) coherent buffer VsStaticVisibility {
    uvec4 vs_visibility[];
};
uint vs_hash(uint h)
{
    h ^= h>>16u;
    h *= 0x7feb352du;
    h ^= h>>15u;
    h *= 0x846ca68bu;
    return h^(h>>16u);
}
bool vs_static_visible_hit(vec3 origin,vec3 ray,float distance,uint light)
{
    uint count=uint(vs_visibility.length());
    if(pc.light_count.z<0.0 || count==0u || distance<=0.001)
        return static_hit(origin,ray,distance,light);

    // One-centimeter spatial cells strongly limit inter-pixel shadow leaks.
    // Two independent keys distinguish collisions in the open-address cache.
    ivec3 cell=ivec3(floor(origin*100.0));
    uint a=vs_hash(uint(cell.x)^vs_hash(uint(cell.y))^
                   (light*0x9e3779b9u));
    uint b=vs_hash(uint(cell.z)^vs_hash(uint(cell.y))^
                   (uint(cell.x)*0x85ebca6bu)^light);
    uint slot=a&(count-1u);
    uint state=vs_visibility[slot].x;
    if(state==2u && vs_visibility[slot].y==a &&
       vs_visibility[slot].z==b)
        return vs_visibility[slot].w!=0u;
    if(state==0u &&
       atomicCompSwap(vs_visibility[slot].x,0u,1u)==0u) {
        bool hit=static_hit(origin,ray,distance,light);
        vs_visibility[slot].y=a;
        vs_visibility[slot].z=b;
        vs_visibility[slot].w=hit?1u:0u;
        memoryBarrierBuffer();
        atomicExchange(vs_visibility[slot].x,2u);
        return hit;
    }
    return static_hit(origin,ray,distance,light);
}
#endif
