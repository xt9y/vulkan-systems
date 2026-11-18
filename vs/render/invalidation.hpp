#pragma once
#include <vs/spatial/world.hpp>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace vs::render {
// Separate GPU consumers can rebuild only the pages their dependency changed.
// No assumption that a scene update requires vk::device::idle().
enum class region_change : std::uint32_t {
    none=0, geometry=1u, transforms=2u, materials=4u,
    collision=8u, lights=16u, shadows=32u, probes=64u,
    optics=128u, all=255u
};
inline region_change operator|(region_change a,region_change b) {
    return static_cast<region_change>(
        static_cast<std::uint32_t>(a)|static_cast<std::uint32_t>(b));
}
inline bool any(region_change value,region_change flag) {
    return (static_cast<std::uint32_t>(value)&
            static_cast<std::uint32_t>(flag))!=0u;
}
struct invalidation {
    ::vs::spatial::region_id id{};
    std::uint64_t revision=0;
    region_change changed=region_change::none;
};
class invalidations {
public:
    void mark(invalidation dirty) {
        auto& current=regions_[dirty.id];
        if(dirty.revision<current.revision)return;
        current.id=dirty.id;
        // Accumulate every unprocessed dependency, even across revisions.
        // A newer region replacement does not imply earlier renderer,
        // collision or lighting invalidations have already been applied.
        current.revision=dirty.revision;
        current.changed=current.changed|dirty.changed;
    }
    // Prioritize nearby work while keeping a hard per-frame budget.
    // Untouched changes remain queued, including their merged change masks.
    std::vector<invalidation> consume(std::size_t budget,
                                     ::vs::spatial::region_id focus) {
        std::vector<invalidation> out;
        out.reserve(regions_.size());
        for(const auto& [id,change]:regions_)out.push_back(change);
        const auto distance=[&](::vs::spatial::region_id id) {
            const auto delta=[](std::int64_t a,std::int64_t b) {
                return a>=b?std::uint64_t(a)-std::uint64_t(b):
                            std::uint64_t(b)-std::uint64_t(a);
            };
            return std::min<std::uint64_t>(4096,delta(id.x,focus.x))+
                   std::min<std::uint64_t>(4096,delta(id.z,focus.z))+
                 2*std::min<std::uint64_t>(4096,delta(id.y,focus.y));
        };
        std::sort(out.begin(),out.end(),[&](const auto& a,const auto& b) {
            const auto x=distance(a.id),y=distance(b.id);
            if(x!=y)return x<y;
            if(a.id.y!=b.id.y)return a.id.y<b.id.y;
            if(a.id.z!=b.id.z)return a.id.z<b.id.z;
            return a.id.x<b.id.x;
        });
        out.resize(std::min(budget,out.size()));
        for(const auto& change:out)regions_.erase(change.id);
        return out;
    }
    std::vector<invalidation> consume() {
        return consume(regions_.size(),{});
    }
    std::size_t size() const noexcept{return regions_.size();}
    bool empty() const noexcept{return regions_.empty();}
    void forget(::vs::spatial::region_id id){regions_.erase(id);}
    void clear() {regions_.clear();}
private:
    std::unordered_map<::vs::spatial::region_id,invalidation,
                       ::vs::spatial::region_hash> regions_;
};
} // namespace vs::render
