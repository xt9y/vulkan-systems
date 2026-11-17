#ifndef VS_GLASS_SEGMENT_GLSL
#define VS_GLASS_SEGMENT_GLSL
// Exact secondary-ray transmission through the same glass BVH used when
// baking point-light directional maps. RGB filters are precomputed on CPU.
struct VsGlassNode {
    vec3 minimum;uint next;
    vec3 maximum;uint first_and_count;
};
#ifndef VS_GLASS_VERTEX_BINDING
#define VS_GLASS_VERTEX_BINDING 7
#endif
#ifndef VS_GLASS_NODE_BINDING
#define VS_GLASS_NODE_BINDING 8
#endif
#ifndef VS_GLASS_FILTER_BINDING
#define VS_GLASS_FILTER_BINDING 9
#endif
layout(std430,set=0,binding=VS_GLASS_VERTEX_BINDING) readonly buffer VsGlassVertices {
    vec4 vs_glass_vertices[];
};
layout(std430,set=0,binding=VS_GLASS_NODE_BINDING) readonly buffer VsGlassNodes {
    VsGlassNode vs_glass_nodes[];
};
layout(std430,set=0,binding=VS_GLASS_FILTER_BINDING) readonly buffer VsGlassFilters {
    vec4 vs_glass_filters[];
};

bool vs_glass_box(vec3 origin,vec3 inv_dir,float limit,VsGlassNode box)
{
    vec3 a=(box.minimum-origin)*inv_dir;
    vec3 b=(box.maximum-origin)*inv_dir;
    vec3 lo=min(a,b),hi=max(a,b);
    return max(max(lo.x,lo.y),max(lo.z,0.0)) <=
           min(min(hi.x,hi.y),min(hi.z,limit));
}
float vs_glass_hit(vec3 eye,vec3 ray,float limit,uint index)
{
    vec3 a=vs_glass_vertices[index].xyz;
    vec3 u=vs_glass_vertices[index+1u].xyz;
    vec3 v=vs_glass_vertices[index+2u].xyz;
    vec3 p=cross(ray,v);
    float det=dot(u,p);
    if(abs(det)<1e-7)return limit;
    float inv_det=1.0/det;
    vec3 t=eye-a;
    float x=dot(t,p)*inv_det;
    if(x<0.0 || x>1.0)return limit;
    vec3 q=cross(t,u);
    float y=dot(ray,q)*inv_det;
    if(y<0.0 || x+y>1.0)return limit;
    float distance=dot(v,q)*inv_det;
    return distance>0.001 && distance<limit?distance:limit;
}
// Cap each segment at the nearest eight interfaces. Sorting also suppresses
// duplicate intersections along shared triangle edges; this is essential for
// stable, single-pixel silhouettes on triangulated glass panes.
vec3 vs_glass_segment(vec3 eye,vec3 ray,float distance,uint count)
{
    uint total=min(count,uint(vs_glass_nodes.length()));
    if(total==0u || distance<=0.001)return vec3(1.0);
    vec3 safe=vec3(
        abs(ray.x)<1e-7?(ray.x<0.0?-1e-7:1e-7):ray.x,
        abs(ray.y)<1e-7?(ray.y<0.0?-1e-7:1e-7):ray.y,
        abs(ray.z)<1e-7?(ray.z<0.0?-1e-7:1e-7):ray.z);
    vec3 inverse=1.0/safe;
    if(!vs_glass_box(eye,inverse,distance,vs_glass_nodes[0]))
        return vec3(1.0);

    const uint max_hits=8u;
    float hit_distance[8];
    uint hit_material[8];
    uint hits=0u;
    uint index=0u;
    while(index<total) {
        VsGlassNode node=vs_glass_nodes[index];
        if(!vs_glass_box(eye,inverse,distance,node)) {
            index=max(node.next,index+1u);
            continue;
        }
        uint num=node.first_and_count&15u;
        if(num!=0u) {
            uint first=node.first_and_count>>4u;
            uint end=min(first+num,uint(vs_glass_vertices.length())/3u);
            for(uint i=first;i<end;++i) {
                uint base=i*3u;
                float t=vs_glass_hit(eye,ray,distance,base);
                if(t>=distance)continue;
                bool duplicate=false;
                for(uint j=0u;j<hits;++j)
                    if(abs(hit_distance[j]-t)<.0002)
                        duplicate=true;
                if(duplicate)continue;
                uint slot=0u;
                while(slot<hits && hit_distance[slot]<t)++slot;
                if(slot>=max_hits)continue;
                uint last=min(hits,max_hits-1u);
                for(uint j=last;j>slot;--j) {
                    hit_distance[j]=hit_distance[j-1u];
                    hit_material[j]=hit_material[j-1u];
                }
                hit_distance[slot]=t;
                hit_material[slot]=uint(vs_glass_vertices[base].w);
                hits=min(hits+1u,max_hits);
            }
        }
        ++index;
    }
    vec3 transmittance=vec3(1.0);
    for(uint j=0u;j<hits;++j) {
        uint material=hit_material[j];
        if(material<uint(vs_glass_filters.length()))
            transmittance*=vs_glass_filters[material].xyz;
    }
    return transmittance;
}

#ifdef VS_EXACT_DIRECT_TRANSMISSION
// Direct visible shading traces geometry, not the low-resolution directional
// atlas. The atlas header is used ONLY to cull lights that cannot hit glass.
// The receiver bias excludes the receiver's own surface at the last endpoint.
// Direct lighting already knows the normalized light ray and distance.
// Reusing them avoids another length and normalization per shaded pixel.
vec3 vs_exact_light_transmission_ray(uint light,vec3 emitter,
                                      vec3 from_emitter_normalized,
                                      float distance,float receiver_bias)
{
    if(light>=64u || vs_transmission.map_index[light]==0u ||
       distance<=receiver_bias+.002)return vec3(1.0);
    float source_bias=min(.002,distance*.1);
    return vs_glass_segment(
        emitter+from_emitter_normalized*source_bias,
        from_emitter_normalized,distance-source_bias-receiver_bias,
        uint(vs_glass_nodes.length()));
}
vec3 vs_exact_light_transmission(uint light,vec3 emitter,vec3 receiver,
                                 float receiver_bias)
{
    if(light>=64u || vs_transmission.map_index[light]==0u)
        return vec3(1.0);
    vec3 delta=receiver-emitter;
    float distance=length(delta);
    if(distance<=receiver_bias+.002)return vec3(1.0);
    return vs_exact_light_transmission_ray(
        light,emitter,delta/distance,distance,receiver_bias);
}
#endif
#endif
