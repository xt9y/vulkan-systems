#version 450
// Read the same buffers PCloud uses for real shadow tests and GI.
layout(set=0,binding=0) uniform sampler3D grid_texture;
struct CloudPoint { vec4 position,normal,radiance; };
layout(std430,set=0,binding=1) readonly buffer Points { CloudPoint points[]; };
struct Node { vec3 minimum; uint next; vec3 maximum; uint first_and_count; };
layout(std430,set=0,binding=2) readonly buffer Nodes { Node nodes[]; };
layout(push_constant) uniform View {
    mat4 projection;
    vec4 grid_origin_spacing;
    uvec4 meta;          // mode, grid width, height, depth
    vec4 viewport;       // width, height, shared BVH nodes, unused
} pc;
layout(location=0) out vec3 ink;
layout(location=1) out vec2 disc;
layout(location=2) flat out uint kind;
layout(location=3) out float opacity;

// Explicit twelve unique cube edges. Vertex 0..23 addresses them by
// index; the 3D axis varies in the edge_pairs lookup below.
const uvec2 edge_pairs[12]=uvec2[12](
    uvec2(0,1),uvec2(1,3),uvec2(3,2),uvec2(2,0),
    uvec2(4,5),uvec2(5,7),uvec2(7,6),uvec2(6,4),
    uvec2(0,4),uvec2(1,5),uvec2(2,6),uvec2(3,7)
);
vec3 corner(vec3 lo,vec3 hi,uint index)
{
    return vec3((index&1u)!=0u?hi.x:lo.x,
                (index&2u)!=0u?hi.y:lo.y,
                (index&4u)!=0u?hi.z:lo.z);
}
void main()
{
    kind=pc.meta.x;
    disc=vec2(0.0);
    opacity=.94;
    if(kind==1u) {
        Node node=nodes[gl_InstanceIndex];
        uint corner_index=(gl_VertexIndex&1)!=0?
            edge_pairs[gl_VertexIndex/2].y:
            edge_pairs[gl_VertexIndex/2].x;
        vec3 pos=corner(node.minimum,node.maximum,corner_index);
        gl_Position=pc.projection*vec4(pos,1.0);
        // Green represents the useful ray/triangle leaves. Red is the
        // traversed internal hierarchy; appended light-local leaves
        // have a brighter green to distinguish specialized acceleration.
        bool leaf=(node.first_and_count&15u)!=0u;
        bool local=float(gl_InstanceIndex)>=pc.viewport.z;
        ink=leaf?(local?vec3(.25,1.0,.45):vec3(.19,.82,.37)):
                 vec3(.93,.17,.22);
        // Many nested internal AABBs overlap on screen. Keep all boxes,
        // but give the useful leaves priority instead of painting them red.
        opacity=leaf?(local?.88:.68):.085;
        return;
    }
    const vec2 offsets[6]=vec2[6](
        vec2(-1,-1),vec2(1,-1),vec2(1,1),
        vec2(-1,-1),vec2(1,1),vec2(-1,1)
    );
    vec3 position;
    vec3 radiance;
    if(kind==2u) {
        uint index=uint(gl_InstanceIndex),wx=pc.meta.y,wy=pc.meta.z;
        ivec3 cell=ivec3(index%wx,(index/wx)%wy,index/(wx*wy));
        radiance=texelFetch(grid_texture,cell,0).rgb;
        position=pc.grid_origin_spacing.xyz+
                 vec3(cell)*pc.grid_origin_spacing.w;
        if(dot(radiance,radiance)<1e-8) {
            gl_Position=vec4(3,3,3,1);
            return;
        }
    } else {
        CloudPoint point=points[gl_InstanceIndex];
        position=point.position.xyz;
        radiance=point.radiance.rgb;
    }
    vec4 projected=pc.projection*vec4(position,1.0);
    if(projected.w<=0.01) {
        gl_Position=vec4(3,3,3,1);
        return;
    }
    // Dark point-cloud samples still receive a visible neutral locator;
    // completely empty grid probes are omitted above rather than black.
    vec3 mapped=max(radiance,vec3(0.0));
    ink=kind==2u?
        max(pow(mapped/(vec3(1.0)+mapped),vec3(.6)),vec3(.12)):
        mix(vec3(.30,.43,.48),pow(mapped/(vec3(1.0)+mapped),vec3(.65)),
            clamp(length(mapped)*2.0,0.0,1.0));
    disc=offsets[gl_VertexIndex];
    float radius=kind==2u?3.5:2.3;
    gl_Position=projected+vec4(
        disc.x*2.0*radius/pc.viewport.x*projected.w,
        disc.y*2.0*radius/pc.viewport.y*projected.w,0,0);
}
