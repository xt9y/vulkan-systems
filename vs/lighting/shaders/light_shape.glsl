#ifndef VS_LIGHT_SHAPE_GLSL
#define VS_LIGHT_SHAPE_GLSL
// A separate std430 vec4 per scene light. Negative w means omnidirectional;
// 0..1 means cos(half cone angle), with a small smooth penumbra.
#ifndef VS_LIGHT_SHAPE_BINDING
#define VS_LIGHT_SHAPE_BINDING 5
#endif
layout(std430,set=0,binding=VS_LIGHT_SHAPE_BINDING)
readonly buffer VsLightShapes { vec4 vs_light_shapes[]; };
// Call from lighting loops that already have a normalized direction.
// Reject out-of-cone lights before BVH shadow and glass traversal.
float vs_light_cone_direction(uint index,vec3 from_emitter_normalized)
{
    if(index>=uint(vs_light_shapes.length()))return 1.0;
    vec4 cone=vs_light_shapes[index];
    if(cone.w<0.0)return 1.0;
    float angle=dot(from_emitter_normalized,cone.xyz);
    float edge=min(.9999,mix(cone.w,1.0,.15));
    return smoothstep(cone.w,max(cone.w+.0001,edge),angle);
}
float vs_light_cone(uint index,vec3 from_emitter)
{
    if(index>=uint(vs_light_shapes.length()) ||
       vs_light_shapes[index].w<0.0)return 1.0;
    float distance=length(from_emitter);
    return distance<1e-5?1.0:
        vs_light_cone_direction(index,from_emitter/distance);
}
#endif
