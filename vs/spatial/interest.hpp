#pragma once
#include <vs/spatial/world.hpp>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace vs::spatial {
// A deterministic region-interest policy. Produces work requests, never
// generates geometry or performs I/O. Existing requests are retained until
// explicitly removed; callbacks can schedule jobs or network transfers.
class interest {
public:
    struct settings {
        std::uint32_t horizontal=6, vertical=0;
        std::size_t max_requests=16, max_removals=32;
    };
    struct change {
        region_id id{};
        bool load=false;
    };
    std::vector<change> update(region_id center,settings options) {
        if(options.horizontal>32u || options.vertical>8u)
            throw std::invalid_argument("Unbounded region interest radius");
        struct candidate {region_id id; std::uint64_t distance;};
        std::vector<candidate> candidates;
        const auto h=std::int64_t(options.horizontal);
        const auto v=std::int64_t(options.vertical);
        candidates.reserve(std::size_t((2*h+1)*(2*h+1)*(2*v+1)));
        std::unordered_set<region_id,region_hash> desired;
        for(std::int64_t y=-v;y<=v;++y)
        for(std::int64_t z=-h;z<=h;++z)
        for(std::int64_t x=-h;x<=h;++x) {
            region_id id{
                checked_region(static_cast<long double>(center.x)+x),
                checked_region(static_cast<long double>(center.y)+y),
                checked_region(static_cast<long double>(center.z)+z)};
            const auto d=std::uint64_t(x*x+z*z+4*y*y);
            candidates.push_back({id,d});
            desired.emplace(id);
        }
        std::sort(candidates.begin(),candidates.end(),
            [](const auto& a,const auto& b) {
                if(a.distance!=b.distance)return a.distance<b.distance;
                if(a.id.y!=b.id.y)return a.id.y<b.id.y;
                if(a.id.z!=b.id.z)return a.id.z<b.id.z;
                return a.id.x<b.id.x;
            });
        std::vector<change> result;
        std::vector<region_id> evictions;
        for(auto id:requested_)if(!desired.contains(id))evictions.push_back(id);
        std::sort(evictions.begin(),evictions.end(),
            [](region_id a,region_id b) {
                if(a.y!=b.y)return a.y<b.y;
                if(a.z!=b.z)return a.z<b.z;
                return a.x<b.x;
            });
        for(const auto id:evictions) {
            if(result.size()>=options.max_removals)break;
            requested_.erase(id);
            result.push_back({id,false});
        }
        std::size_t queued=0;
        for(const auto& candidate:candidates) {
            if(queued>=options.max_requests)break;
            if(requested_.insert(candidate.id).second) {
                result.push_back({candidate.id,true});
                ++queued;
            }
        }
        return result;
    }
    bool contains(region_id id) const {return requested_.contains(id);}
    std::size_t size() const noexcept {return requested_.size();}
    void clear(){requested_.clear();}
private:
    std::unordered_set<region_id,region_hash> requested_;
};
} // namespace vs::spatial
