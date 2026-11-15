// Shared DISPLAY-SPACE glass tint, used for opaque-world, nested-glass,
// and reflected-camera glass. Not a physically complete volume integral.
vec3 optics_glass_layer(vec3 background,vec3 tint,float pass,float solid)
{
    vec3 pigment=clamp(tint,vec3(0.0),vec3(1.0));
    float density=clamp(1.0-dot(pigment,vec3(.2126,.7152,.0722)),
                        0.0,1.0);
    vec3 attenuation=exp(-(solid>.5?0.85:0.25)*(vec3(1.0)-pigment));
    vec3 transmitted=mix(pigment,background*attenuation,
                         clamp(pass,0.0,1.0));
    return mix(transmitted,pigment,0.22*density);
}
