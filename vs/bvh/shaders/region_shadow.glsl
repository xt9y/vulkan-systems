#ifndef VS_REGION_SHADOW_GLSL
#define VS_REGION_SHADOW_GLSL
// Region pages have stable per-image buffer storage. The header's first
// span.x is the active count; each following record refers to one region.
// Node next and triangle indices remain relative to the region's own BVH.
struct VsRegionRecord {
    vec4 minimum,maximum;
    uvec4 span;
};
layout(std430,set=0,binding=19) readonly buffer VsRegionRecords {
    VsRegionRecord vs_region_records[];
};
layout(std430,set=0,binding=20) readonly buffer VsRegionVertices {
    vec4 vs_region_vertices[];
};
layout(std430,set=0,binding=21) readonly buffer VsRegionNodes {
    ShadowNode vs_region_nodes[];
};
bool vs_region_static_hit(vec3 origin,vec3 ray,float distance)
{
    if(vs_region_records.length()<2 || distance<=0.001)return false;
    uint count=min(vs_region_records[0].span.x,
                   uint(vs_region_records.length()-1));
    if(count==0u)return false;
    vec3 safe=vec3(
        abs(ray.x)<1e-7?(ray.x<0.0?-1e-7:1e-7):ray.x,
        abs(ray.y)<1e-7?(ray.y<0.0?-1e-7:1e-7):ray.y,
        abs(ray.z)<1e-7?(ray.z<0.0?-1e-7:1e-7):ray.z);
    vec3 inv_dir=1.0/safe;
    for(uint page=1u;page<=count;++page) {
        VsRegionRecord region=vs_region_records[page];
        if(region.span.z==0u)continue;
        ShadowNode bounds;
        bounds.minimum=region.minimum.xyz;
        bounds.maximum=region.maximum.xyz;
        bounds.next=0u;bounds.first_and_count=0u;
        if(!shadow_box(origin,inv_dir,distance,bounds))continue;
        uint at=0u;
        uint end=min(region.span.z,
                     uint(vs_region_nodes.length())-region.span.y);
        while(at<end) {
            ShadowNode node=vs_region_nodes[region.span.y+at];
            if(!shadow_box(origin,inv_dir,distance,node)) {
                at=max(node.next,at+1u);
                continue;
            }
            uint triangles=node.first_and_count&15u;
            if(triangles>0u) {
                uint first=node.first_and_count>>4u;
                uint last=min(first+triangles,region.span.w/3u);
                for(uint i=first;i<last;++i) {
                    uint v=region.span.x+i*3u;
                    if(v+2u>=uint(vs_region_vertices.length()))break;
                    if(triangle_hit(origin,ray,distance,
                        vs_region_vertices[v].xyz,
                        vs_region_vertices[v+1u].xyz,
                        vs_region_vertices[v+2u].xyz))
                        return true;
                }
            }
            ++at;
        }
    }
    return false;
}
#endif
