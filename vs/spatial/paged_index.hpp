#pragma once
#include <vs/spatial/world.hpp>
#include <vs/spatial/index.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace vs::spatial {
// Region-local BVHs indexed by their occupied world cells. Replacing one
// region never modifies another region's BVH. Queries visit nearby cells
// instead of testing every loaded region in the world.
class paged_index {
    static constexpr std::size_t max_page_cells=64;
    static constexpr std::size_t max_query_cells=4096;
    struct page {
        index tree;
        bounds extent{};
        std::vector<region_id> occupied;
        bool wide=false;
    };
    std::unordered_map<region_id,page,region_hash> pages_;
    std::unordered_map<region_id,std::unordered_set<region_id,region_hash>,
                       region_hash> cells_;
    std::unordered_set<region_id,region_hash> wide_;

    static region_id cell(region_id origin,vec3 position) {
        return normalize({origin,{double(position.x),double(position.y),
                                  double(position.z)}}).region;
    }
    template<class Visit>
    static bool cells(region_id lo,region_id hi,std::size_t limit,Visit&& visit) {
        if(lo.x>hi.x || lo.y>hi.y || lo.z>hi.z)return false;
        const auto dx=std::uint64_t(hi.x)-std::uint64_t(lo.x);
        const auto dy=std::uint64_t(hi.y)-std::uint64_t(lo.y);
        const auto dz=std::uint64_t(hi.z)-std::uint64_t(lo.z);
        if(dx>=limit || dy>=limit || dz>=limit ||
           (dx+1u)>(limit/(dy+1u))/(dz+1u))return false;
        for(auto z=lo.z;;++z) {
            for(auto y=lo.y;;++y) {
                for(auto x=lo.x;;++x) {
                    visit(region_id{x,y,z});
                    if(x==hi.x)break;
                }
                if(y==hi.y)break;
            }
            if(z==hi.z)break;
        }
        return true;
    }
    static bool offset(std::int64_t origin,std::int64_t page,float& value) {
        const bool positive=origin>=page;
        const auto gap=positive?
            std::uint64_t(origin)-std::uint64_t(page):
            std::uint64_t(page)-std::uint64_t(origin);
        if(gap>10000000u)return false;
        value=(positive?1.f:-1.f)*float(double(gap)*region_extent);
        return true;
    }
public:
    void set(region_id id,std::span<const bounds> boxes) {
        if(boxes.empty()){erase(id);return;}
        page fresh{};
        fresh.extent=boxes.front();
        for(const auto& b:boxes) {
            if(b.min.x>b.max.x || b.min.y>b.max.y || b.min.z>b.max.z)
                throw std::invalid_argument("Invalid region bounds");
            fresh.extent.min={
                std::min(fresh.extent.min.x,b.min.x),
                std::min(fresh.extent.min.y,b.min.y),
                std::min(fresh.extent.min.z,b.min.z)};
            fresh.extent.max={
                std::max(fresh.extent.max.x,b.max.x),
                std::max(fresh.extent.max.y,b.max.y),
                std::max(fresh.extent.max.z,b.max.z)};
        }
        fresh.tree.rebuild(std::vector<bounds>(boxes.begin(),boxes.end()));
        fresh.wide=!cells(cell(id,fresh.extent.min),
                          cell(id,fresh.extent.max),max_page_cells,
                          [&](region_id key){fresh.occupied.push_back(key);});
        erase(id);
        auto [it,inserted]=pages_.emplace(id,std::move(fresh));
        const auto& item=it->second;
        if(item.wide)wide_.insert(id);
        else for(auto key:item.occupied)cells_[key].insert(id);
    }
    bool erase(region_id id) {
        const auto it=pages_.find(id);
        if(it==pages_.end())return false;
        if(it->second.wide)wide_.erase(id);
        else for(auto key:it->second.occupied) {
            auto found=cells_.find(key);
            if(found==cells_.end())continue;
            found->second.erase(id);
            if(found->second.empty())cells_.erase(found);
        }
        pages_.erase(it);
        return true;
    }
    void clear(){pages_.clear();cells_.clear();wide_.clear();}
    std::size_t size() const noexcept{return pages_.size();}

    template<class Visit>
    void query(region_id origin,const bounds& relative,Visit&& visit) const {
        if(relative.min.x>relative.max.x ||
           relative.min.y>relative.max.y ||
           relative.min.z>relative.max.z)return;
        std::unordered_set<region_id,region_hash> candidates=wide_;
        if(!cells(cell(origin,relative.min),cell(origin,relative.max),
                  max_query_cells,[&](region_id key) {
                      if(const auto it=cells_.find(key);it!=cells_.end())
                          candidates.insert(it->second.begin(),it->second.end());
                  })) {
            // Extremely large queries fall back to visiting every page.
            for(const auto& [id,item]:pages_)candidates.insert(id);
        }
        std::vector<region_id> ordered(candidates.begin(),candidates.end());
        std::sort(ordered.begin(),ordered.end(),[](region_id a,region_id b) {
            if(a.y!=b.y)return a.y<b.y;
            if(a.z!=b.z)return a.z<b.z;
            return a.x<b.x;
        });
        for(auto id:ordered) {
            const auto it=pages_.find(id);
            if(it==pages_.end())continue;
            float dx=0.f,dy=0.f,dz=0.f;
            if(!offset(origin.x,id.x,dx) ||
               !offset(origin.y,id.y,dy) ||
               !offset(origin.z,id.z,dz))continue;
            const vec3 move{dx,dy,dz};
            const bounds local{relative.min+move,relative.max+move};
            if(!overlaps(it->second.extent,local))continue;
            it->second.tree.query(local,[&](std::uint32_t primitive) {
                visit(id,primitive);
            });
        }
    }
};
} // namespace vs::spatial
