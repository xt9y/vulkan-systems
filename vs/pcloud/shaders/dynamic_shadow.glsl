#ifndef VS_DYNAMIC_SHADOW_GLSL
#define VS_DYNAMIC_SHADOW_GLSL

layout(std430,set=0,binding=VS_DYNAMIC_VERTEX_BINDING) readonly buffer VsDynamicVertices {
    vec4 vs_dynamic_vertices[];
};
layout(std430,set=0,binding=VS_DYNAMIC_NODE_BINDING) readonly buffer VsDynamicNodes {
    uvec4 vs_dynamic_counts;
    ShadowNode vs_dynamic_nodes[];
};
bool vs_dynamic_triangle_hit(vec3 origin,vec3 ray,float limit,
                             vec3 a,vec3 e1,vec3 e2)
{
    vec3 p=cross(ray,e2);
    float det=dot(e1,p);
    if(abs(det)<.00001)return false;
    float inv=1.0/det;
    vec3 t=origin-a;
    float u=dot(t,p)*inv;
    if(u<0.0 || u>1.0)return false;
    vec3 q=cross(t,e1);
    float v=dot(ray,q)*inv;
    if(v<0.0 || u+v>1.0)return false;
    float distance=dot(e2,q)*inv;
    return distance>.001 && distance<limit;
}

vec3 vs_shadow_inverse_direction(vec3 ray)
{
    vec3 safe=vec3(
        abs(ray.x)<1e-7?(ray.x<0.0?-1e-7:1e-7):ray.x,
        abs(ray.y)<1e-7?(ray.y<0.0?-1e-7:1e-7):ray.y,
        abs(ray.z)<1e-7?(ray.z<0.0?-1e-7:1e-7):ray.z);
    return 1.0/safe;
}
bool vs_dynamic_hit(vec3 origin,vec3 ray,float distance,
                    vec3 inverse_direction)
{
    uint total=vs_dynamic_counts.x;
    uint vertex_count=vs_dynamic_counts.y;
    if(total==0u || vertex_count==0u || distance<=.001)return false;
    uint index=0u;
    while(index<total) {
        ShadowNode node=vs_dynamic_nodes[index];
        if(!shadow_box(origin,inverse_direction,distance,node)) {
            index=node.next;
            continue;
        }
        uint count=node.first_and_count&15u;
        if(count!=0u) {
            uint first=node.first_and_count>>4u;
            uint end=first+count;
            for(uint i=first;i<end;++i) {
                uint v=i*3u;
#ifdef VS_DYNAMIC_TRIANGLES_ONLY
                if(vs_dynamic_triangle_hit(origin,ray,distance,
                    vs_dynamic_vertices[v].xyz,
                    vs_dynamic_vertices[v+1u].xyz,
                    vs_dynamic_vertices[v+2u].xyz))
                    return true;
#else
                vec4 primitive=vs_dynamic_vertices[v];
                if(primitive.w<0.0) {
                    // The same BVH stores animated bone spheres and triangles.
                    vec3 delta=origin-primitive.xyz;
                    float radius=-primitive.w;
                    float c=dot(delta,delta)-radius*radius;
                    // A receiver inside a coarse sphere must not self-shadow.
                    if(c>0.0) {
                        float half_b=dot(delta,ray);
                        float discriminant=half_b*half_b-c;
                        if(half_b<0.0 && discriminant>=0.0) {
                            float entry=-half_b-sqrt(discriminant);
                            if(entry>0.001 && entry<distance)return true;
                        }
                    }
                } else if(vs_dynamic_triangle_hit(origin,ray,distance,
                    primitive.xyz,
                    vs_dynamic_vertices[v+1u].xyz,
                    vs_dynamic_vertices[v+2u].xyz))
                    return true;
#endif
            }
        }
        ++index;
    }
    return false;
}

bool vs_dynamic_hit(vec3 origin,vec3 ray,float distance)
{
    if(vs_dynamic_counts.x==0u || vs_dynamic_counts.y==0u ||
       distance<=.001)return false;
    return vs_dynamic_hit(origin,ray,distance,
                          vs_shadow_inverse_direction(ray));
}

#endif
