#pragma once
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>

namespace vs::spatial {
// Region addresses are exact even when world coordinates exceed float precision.
// Every region contains a fixed, 256-metre local coordinate space.
inline constexpr double region_extent = 256.0;
struct region_id {
    std::int64_t x=0,y=0,z=0;
    bool operator==(const region_id&) const = default;
};
struct region_hash {
    std::size_t operator()(region_id id) const noexcept {
        auto mix=[](std::uint64_t v) {
            v^=v>>30;v*=0xbf58476d1ce4e5b9ull;
            v^=v>>27;v*=0x94d049bb133111ebull;
            return v^(v>>31);
        };
        const auto a=mix(static_cast<std::uint64_t>(id.x));
        const auto b=mix(static_cast<std::uint64_t>(id.y));
        const auto c=mix(static_cast<std::uint64_t>(id.z));
        return static_cast<std::size_t>(mix(a^b^(c<<1)));
    }
};
struct world_position {
    region_id region{};
    std::array<double,3> local{};
};
inline std::int64_t checked_region(long double value) {
    if(!std::isfinite(value) ||
       value < static_cast<long double>(std::numeric_limits<std::int64_t>::min()) ||
       value >= -static_cast<long double>(std::numeric_limits<std::int64_t>::min()))
        throw std::out_of_range("World region coordinate overflow");
    return static_cast<std::int64_t>(value);
}
inline world_position normalize(world_position value) {
    auto coordinate=[&](std::int64_t& region,double& local) {
        if(!std::isfinite(local))throw std::invalid_argument("Invalid world position");
        const auto delta=std::floor(static_cast<long double>(local)/region_extent);
        region=checked_region(static_cast<long double>(region)+delta);
        local-=static_cast<double>(delta*region_extent);
        // Compensate for a rounded local result at the tile boundary.
        if(local<0.0) {
            region=checked_region(static_cast<long double>(region)-1.0L);
            local+=region_extent;
        } else if(local>=region_extent) {
            region=checked_region(static_cast<long double>(region)+1.0L);
            local-=region_extent;
        }
    };
    coordinate(value.region.x,value.local[0]);
    coordinate(value.region.y,value.local[1]);
    coordinate(value.region.z,value.local[2]);
    return value;
}
inline world_position from_world(std::array<double,3> absolute) {
    world_position result{};
    result.local=absolute;
    return normalize(result);
}
inline world_position translated(world_position position,std::array<double,3> delta) {
    for(int i=0;i<3;++i)position.local[i]+=delta[i];
    return normalize(position);
}
inline std::array<float,3> camera_relative(const world_position& position,
                                            const world_position& origin) {
    const std::array<long double,3> region_delta{
        static_cast<long double>(position.region.x)-origin.region.x,
        static_cast<long double>(position.region.y)-origin.region.y,
        static_cast<long double>(position.region.z)-origin.region.z
    };
    std::array<float,3> result{};
    for(int i=0;i<3;++i) {
        const auto value=region_delta[i]*region_extent+
                         static_cast<long double>(position.local[i])-origin.local[i];
        if(!std::isfinite(value) ||
           std::abs(value)>std::numeric_limits<float>::max())
            throw std::out_of_range("Camera relative offset overflow");
        result[i]=static_cast<float>(value);
    }
    return result;
}
} // namespace vs::spatial
