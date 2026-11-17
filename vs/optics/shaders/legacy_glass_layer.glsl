// Shared DISPLAY-SPACE glass tint, used for opaque-world, nested-glass,
// and reflected-camera glass. Not a physically complete volume integral.
vec3 optics_glass_layer(vec3 background,vec3 tint,float pass,float solid)
{
    vec3 pigment=clamp(tint,vec3(0.0),vec3(1.0));
    // Glass is not emissive. Color comes from *actual radiance behind it*
    // and the separately evaluated PBR surface response. A dark background
    // behind unlit glass must remain dark, regardless of its pigment.
    vec3 attenuation=exp(-(solid>.5?1.10:0.35)*(vec3(1.0)-pigment));
    return background*attenuation*clamp(pass,0.0,1.0);
}

 // Match the primary glass normal/Fresnel and bounded screen refraction
 // in secondary views. These values are evaluated only on rasterized
 // glass pixels; there is no extra ray-scene intersection.
struct optics_glass_profile {
    vec2 shift;
    float fresnel;
};
optics_glass_profile optics_glass_shape(vec4 encoded_normal,vec3 incoming,
                                        vec3 right,vec3 up,vec2 half_fov,
                                        float ior,float solid)
{
    vec3 n=normalize(encoded_normal.rgb*2.0-1.0);
    if(dot(n,incoming)>0.0)n=-n;
    float cosine=clamp(dot(-incoming,n),0.0,1.0);
    float f0=(ior-1.0)/(ior+1.0);
    f0*=f0;
    float edge=1.0-cosine;
    optics_glass_profile out_shape;
    float edge2=edge*edge;
    out_shape.fresnel=f0+(1.0-f0)*edge2*edge2*edge;
    out_shape.shift=vec2(0.0);
    if(solid>.5) {
        vec3 ray=refract(incoming,n,1.0/ior);
        if(dot(ray,ray)>1e-6) {
            vec3 bend=normalize(ray)-incoming;
            out_shape.shift=clamp(vec2(
                dot(bend,right)/max(half_fov.x,.001),
               -dot(bend,up)/max(half_fov.y,.001))*.08,
                vec2(-.02),vec2(.02));
        }
    }
    return out_shape;
}
vec3 optics_glass_shaped(vec3 through,vec3 surface,
                         vec3 tint,float pass,float solid,float fresnel)
{
    return mix(optics_glass_layer(through,tint,pass,solid),
               surface,fresnel);
}
