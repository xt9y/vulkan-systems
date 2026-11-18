#pragma once
#include <vs/spatial/world.hpp>
#include <vs/spatial/index.hpp>
#include <array>
#include <algorithm>
#include <cmath>
#include <span>
#include <unordered_map>
#include <vector>

namespace vs::spatial {
// Each region owns its own BVH; replacing/removing a region never rebuilds
// other regions. Bounds supplied to set() are local to that region's origin.
// A query is expressed relative to any world region and visits (id, primitive).
class paged_index {
    struct page {
        std::vector<bounds> primitives;
        index tree;
        bounds extent{};
    };
public:
    void set(region_id id,std::span<const bounds> boxes) {
        if(boxes.empty()){erase(id);return;}
        page fresh;
        fresh.primitives.assign(boxes.begin(),boxes.end());
        fresh.extent=boxes.front();
        for(const auto& b:boxes) {
            fresh.extent.min={
                std::min(fresh.extent.min.x,b.min.x),
                std::min(fresh.extent.min.y,b.min.y),
                std::min(fresh.extent.min.z,b.min.z)};
            fresh.extent.max={
                std::max(fresh.extent.max.x,b.max.x),
                std::max(fresh.extent.max.y,b.max.y),
                std::max(fresh.extent.max.z,b.max.z)};
        }
        fresh.tree.rebuild(fresh.primitives);
        pages_.insert_or_assign(id,std::move(fresh));
    }
    bool erase(region_id id) {return pages_.erase(id)!=0;}
    void clear(){pages_.clear();}
    std::size_t size() const noexcept {return pages_.size();}
    template<class Visit>
    void query(region_id origin,const bounds& relative,Visit&& visit) const {
        for(const auto& [id,page]:pages_) {
            const auto dx=(static_cast<long double>(origin.x)-id.x)*region_extent;
            const auto dy=(static_cast<long double>(origin.y)-id.y)*region_extent;
            const auto dz=(static_cast<long double>(origin.z)-id.z)*region_extent;
            if(std::abs(dx)>1.e12L || std::abs(dy)>1.e12L || std::abs(dz)>1.e12L)
                continue;
            const vec3 offset{static_cast<float>(dx),static_cast<float>(dy),
                              static_cast<float>(dz)};
            const bounds local{relative.min+offset,relative.max+offset};
            if(!overlaps(page.extent,local))continue;
            page.tree.query(local,[&](std::uint32_t primitive) {
                visit(id,primitive);
            });
        }
    }
private:
    std::unordered_map<region_id,page,region_hash> pages_;
};
} // namespace vs::spatial
