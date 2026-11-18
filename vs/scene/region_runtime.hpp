#pragma once
#include <vs/scene/regions.hpp>
#include <vs/render/invalidation.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace vs::scene {
// Application-controlled region scene. Assets live outside this object and may
// be referenced by many fragments; a replacement touches one region only.
// The consumer drains invalidations and updates its own GPU/physics structures.
class region_runtime {
public:
    bool publish(::vs::spatial::region_id id,std::uint64_t revision,
                 std::shared_ptr<const region_fragment> fragment,
                 ::vs::render::region_change affected=
                     ::vs::render::region_change::all) {
        if(!regions_.publish(id,revision,std::move(fragment)))return false;
        dirty_.mark({id,revision,affected});
        return true;
    }
    bool remove(::vs::spatial::region_id id,std::uint64_t revision) {
        if(!regions_.remove(id,revision))return false;
        dirty_.mark({id,revision,::vs::render::region_change::all});
        return true;
    }
    const region_catalog& regions() const noexcept {return regions_;}
    bool dirty() const noexcept {return !dirty_.empty();}
    std::vector<::vs::render::invalidation> take_changes() {
        return dirty_.consume();
    }
    std::vector<::vs::render::invalidation> take_changes(
        std::size_t budget,::vs::spatial::region_id focus) {
        return dirty_.consume(budget,focus);
    }
    void clear() {regions_.clear();dirty_.clear();}
private:
    region_catalog regions_;
    ::vs::render::invalidations dirty_;
};
} // namespace vs::scene
