// Include in your skinned fragment shader. Bind one actor's bld::render::wound
// descriptor at set 0, binding 14. Data is packed R4 coverage/G4 wetness.
layout(std430, set = 0, binding = 14) readonly buffer BloodWounds
{
    uint blood_pixels[];
};

vec2 bld_wound_texel(ivec2 xy, int width)
{
    xy = clamp(xy, ivec2(0), ivec2(width - 1));
    uint pixel = uint(xy.y * width + xy.x);
    uint value = (blood_pixels[pixel / 4u] >> ((pixel & 3u) * 8u)) & 255u;
    return vec2(float(value >> 4u), float(value & 15u)) / 15.0;
}

vec2 bld_wound_sample(vec2 uv, int width)
{
    vec2 p = clamp(uv, vec2(0), vec2(1)) * float(width - 1);
    ivec2 origin = ivec2(floor(p));
    vec2 t = fract(p);
    return mix(
        mix(bld_wound_texel(origin, width),
            bld_wound_texel(origin + ivec2(1, 0), width), t.x),
        mix(bld_wound_texel(origin + ivec2(0, 1), width),
            bld_wound_texel(origin + ivec2(1, 1), width), t.x),
        t.y
    );
}

void bld_wound_material(vec2 uv, int width,
                        inout vec3 albedo, inout float roughness)
{
    vec2 wound = bld_wound_sample(uv, width);
    float coverage = wound.x;
    float wetness = wound.y;
    vec3 dried = vec3(.05, .008, .006);
    vec3 fresh = vec3(.17, .006, .011);
    albedo = mix(albedo, mix(dried, fresh, wetness), coverage);
    roughness = mix(roughness, mix(.78, .13, wetness), coverage);
}
