#pragma once
#include <fld/fld.hpp>
#include <cstdint>
#include <functional>

namespace bld
{
    using vec2=fld::vec2;
    using vec3=fld::vec3;
    using vec4=fld::vec4;
    using mat4=fld::mat4;
    using instance_id=std::uint32_t;

    enum class substrate : std::uint8_t {
        skin, textile, porous, painted, metal, glass
    };

    struct impact
    {
        std::uint64_t id=0;
        std::uint32_t source=0, target=0, tick=0, seed=1;
        vec3 position{}, normal{0,1,0}, direction{0,0,1};
        vec2 uv{};
        bool uv_valid=false;
        float amount_ml=8.f;   // source volume in millilitres
        float velocity=5.f;    // impact ejecta speed, metres/second
    };

    struct options
    {
        fld::options fluid{};
        std::uint32_t wound_resolution=256;
        std::uint32_t max_marks=512;
        std::uint32_t max_stains=1024;
        float drying_time=42.f;   // seconds
    };

    using collision_callback=fld::collision_callback;
}
