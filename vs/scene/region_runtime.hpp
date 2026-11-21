#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>
#include <vs/render/invalidation.hpp>
#include <vs/scene/regions.hpp>
#include <vs/spatial/paged_index.hpp>

namespace vs::scene
{
// Application-controlled region scene. Assets live outside this object and may
// be referenced by many fragments; a replacement touches one region only.
// The consumer drains invalidations and updates its own GPU/physics structures.
class region_runtime
{
public:
    bool publish(::vs::spatial::region_id id, std::uint64_t revision,
                 std::shared_ptr<const region_fragment> fragment,
                 ::vs::render::region_change affected =
                     ::vs::render::region_change::all)
    {
        if (!fragment || !revision)
            throw std::invalid_argument("Invalid region fragment or revision");

        if (const auto *current = regions_.get(id);
            current && revision <= current->revision) return false;
        // Region-local collision geometry is prepared independently from
        // the mesh/texture asset store and all other regions.
        std::vector<::vs::spatial::bounds> boxes;
        boxes.reserve(fragment->collision_boxes.size());

        for (const auto &box : fragment->collision_boxes)
            boxes.push_back({box.minimum, box.maximum});
        collision_.set(id, boxes);

        if (!regions_.publish(id, revision, std::move(fragment))) return false;
        dirty_.mark({id, revision, affected});

        return true;
    }

    bool remove(::vs::spatial::region_id id, std::uint64_t revision)
    {
        if (!regions_.remove(id, revision)) return false;
        collision_.erase(id);
        dirty_.mark({id, revision, ::vs::render::region_change::all});

        return true;
    }
    // Returns a region-local box ID; callers resolve material/physics in the
    // authoritative fragment. Neither world coordinates nor mesh copies
    // are required by the broad phase.
    template <class Visit>
    void query_collision(::vs::spatial::region_id query_origin,
                         const ::vs::spatial::bounds &query_box,
                         Visit &&visit) const
    {
        collision_.query(query_origin, query_box,
                         std::forward<Visit>(visit));
    }

    const region_fragment *find(::vs::spatial::region_id id) const
    {
        const auto *entry = regions_.get(id);

        return entry && entry->value ? entry->value.get() : nullptr;
    }
    // Evict an out-of-interest region without an authoritative tombstone:
    // re-entering the region may reuse the same content revision.
    bool release(::vs::spatial::region_id id)
    {
        if (!regions_.forget(id)) return false;
        collision_.erase(id);
        dirty_.forget(id);
        dirty_.mark({id, 1u, ::vs::render::region_change::all});

        return true;
    }

    const region_catalog &regions() const noexcept { return regions_; }
    bool dirty() const noexcept { return !dirty_.empty(); }
    std::vector<::vs::render::invalidation> take_changes()
    {
        return dirty_.consume();
    }

    std::vector<::vs::render::invalidation> take_changes(
        std::size_t budget, ::vs::spatial::region_id focus)
    {
        return dirty_.consume(budget, focus);
    }
    // A network resnapshot discards local revision history but must notify
    // GPU consumers about regions absent from the replacement snapshot.
    // Otherwise old shadow BVHs or resident geometry survive indefinitely.
    void clear_and_invalidate()
    {
        std::vector<::vs::spatial::region_id> resident;
        regions_.each([&](auto id, auto, const auto &)
                      { resident.push_back(id); });
        clear();

        for (const auto id : resident)
            dirty_.mark({id, 1u, ::vs::render::region_change::all});
    }

    void clear()
    {
        regions_.clear();
        collision_.clear();
        dirty_.clear();
    }

private:
    region_catalog regions_;
    ::vs::spatial::paged_index collision_;
    ::vs::render::invalidations dirty_;
};
} // namespace vs::scene
