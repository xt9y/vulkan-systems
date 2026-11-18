#pragma once
#include <vs/spatial/world.hpp>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace vs::render {
// Masks describe independent GPU consumers. No "scene changed -> idle GPU"
// assumption: the frame graph/renderer can rebuild only the affected pages.
enum class region_change : std::uint32_t {
    none=0, geometry=1u, transforms=2u, materials=4u,
    collision=8u, lights=16u, shadows=32u, probes=64u,
    optics=128u, all=255u
};
inline region_change operator|(region_change a,region_change b) {
    return static_cast<region_change>(
        static_cast<std::uint32_t>(a)|static_cast<std::uint32_t>(b));
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
        current.revision=dirty.revision;
        current.changed=current.changed|dirty.changed;
    }
    std::vector<invalidation> consume() {
        std::vector<invalidation> out;
        out.reserve(regions_.size());
        for(const auto& [id,dirty]:regions_)out.push_back(dirty);
        regions_.clear();
        return out;
    }
    bool empty() const noexcept{return regions_.empty();}
    void clear() {regions_.clear();}
private:
    std::unordered_map<::vs::spatial::region_id,invalidation,
                       ::vs::spatial::region_hash> regions_;
};
} // namespace vs::render
