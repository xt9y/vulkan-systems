// Composite the RGBA16F output of bld::render onto your lit GBuffer surface.
// The output RGB is dark linear-space pigment, A is coverage.
void bld_composite_blood(vec4 overlay, inout vec3 albedo,
                         inout float roughness, inout float metallic)
{
    float alpha = clamp(overlay.a, 0.0, 1.0);
    float wetness = clamp((overlay.r - .05) / .12, 0.0, 1.0);
    albedo = mix(albedo, overlay.rgb, alpha);
    roughness = mix(roughness, mix(.78, .13, wetness), alpha);
    metallic = mix(metallic, 0.0, alpha);
}
