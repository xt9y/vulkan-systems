#pragma once
#include <vs/gpu/residency.hpp>
#include <vs/spatial/world.hpp>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace vs::gpu {
// Shared mesh/texture pages may be referenced by multiple spatial regions.
// Region unload only releases ownership; actual GPU destruction happens after
// frame_scheduler::completed_serial() allows an unreferenced page to be evicted.
class region_residency {
public:
    using page_id=residency::page_id;
    explicit region_residency(std::size_t budget):pages_(budget){}

    bool attach(::vs::spatial::region_id region,page_id id,
                std::size_t bytes,std::uint64_t submitted_serial) {
        const auto existing=pages_.find(id);
        if(existing && existing->bytes!=bytes)
            throw std::invalid_argument("Reused GPU page ID has different size");
        if(!existing &&
           !pages_.insert(id,bytes,submitted_serial))return false;
        auto& owned=regions_[region];
        if(owned.insert(id).second)++references_[id];
        pages_.touch(id,submitted_serial);
        pages_.pin(id,true);
        return true;
    }

    void touch(page_id id,std::uint64_t submitted_serial) {
        pages_.touch(id,submitted_serial);
    }
    void detach(::vs::spatial::region_id region,
                std::uint64_t last_use_serial) {
        const auto found=regions_.find(region);
        if(found==regions_.end())return;
        for(auto id:found->second) {
            pages_.touch(id,last_use_serial);
            const auto it=references_.find(id);
            if(it==references_.end())continue;
            if(!--it->second) {
                references_.erase(it);
                pages_.pin(id,false);
            }
        }
        regions_.erase(found);
    }
    // Returns only GPU-safe, unreferenced victims. Callers destroy the actual
    // Vulkan buffers/images and remove their descriptor references.
    std::vector<page_id> trim(std::uint64_t completed_serial,
                              std::size_t target_bytes=0) {
        return pages_.trim(completed_serial,target_bytes);
    }
    std::size_t used() const noexcept{return pages_.used();}
    std::size_t budget() const noexcept{return pages_.budget();}
    std::size_t regions() const noexcept{return regions_.size();}
    const residency::page* find(page_id id) const{return pages_.find(id);}
    // Only legal after device idle and all borrowed GPU handles are released.
    void clear_idle() {
        regions_.clear();
        references_.clear();
        pages_.clear_idle();
    }
private:
    residency pages_;
    std::unordered_map<::vs::spatial::region_id,
                       std::unordered_set<page_id>,
                       ::vs::spatial::region_hash> regions_;
    std::unordered_map<page_id,std::size_t> references_;
};
} // namespace vs::gpu
