// Optional fallback for meshes without useful UV coordinates.
// The receiving-object check is mandatory: never project a blood mark onto
// another surface just because its pixels happen to overlap on screen.
struct BldRigidDeposit
{
    mat4 inverse_world;
    vec4 normal_radius; // xyz world-space normal, w radius (metres)
    vec4 parameters;    // x wetness, y amount, z variation, w receiver ID
};

float bld_rigid_mask(vec3 world,vec3 normal,
                     uint receiving_instance,BldRigidDeposit stain)
{
    if(receiving_instance!=uint(stain.parameters.w+.5))return 0.0;
    vec3 p=(stain.inverse_world*vec4(world,1.0)).xyz;
    if(abs(p.z)>.007)return 0.0;
    if(dot(normal,normalize(stain.normal_radius.xyz))<.65)return 0.0;
    vec2 offset=p.xy/max(stain.normal_radius.w,.0001);
    float r=length(offset);
    float a=atan(offset.y,offset.x);
    float edge=.84+.075*sin(a*7.0+stain.parameters.z)+
               .045*sin(a*13.0-2.0*stain.parameters.z);
    return 1.0-smoothstep(edge-.1,edge+.02,r);
}
