#pragma once
#include <vs/spatial/world.hpp>
#include <algorithm>
#include <cstdint>
#include <deque>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace vs::spatial {
// A versioned, application-owned collection of immutable region payloads.
// No renderer, world generator or networking dependency. Tombstones prevent
// stale asynchronous completions from resurrecting unloaded regions.
template<class Payload> class region_store {
public:
    struct entry {
        std::uint64_t revision=0;
        std::shared_ptr<const Payload> value;
    };
    struct change {
        std::uint64_t serial=0, revision=0;
        region_id id{};
        bool present=false;
    };
    explicit region_store(std::size_t journal_limit=8192)
        :limit_(std::max<std::size_t>(1,journal_limit)) {}

    bool publish(region_id id,std::uint64_t revision,
                 std::shared_ptr<const Payload> payload) {
        if(!revision || !payload)throw std::invalid_argument("Invalid region revision or payload");
        auto& item=items_[id];
        if(revision<=item.revision)return false;
        item={revision,std::move(payload)};
        record({++serial_,revision,id,true});
        return true;
    }
    bool remove(region_id id,std::uint64_t revision) {
        if(!revision)throw std::invalid_argument("Invalid region revision");
        auto& item=items_[id];
        if(revision<=item.revision)return false;
        item={revision,{}};
        record({++serial_,revision,id,false});
        return true;
    }
    const entry* get(region_id id) const {
        const auto it=items_.find(id);
        return it==items_.end()?nullptr:&it->second;
    }
    // Local interest eviction is not an authoritative deletion. Forget
    // the old revision so the same content revision may be streamed again
    // under a newer subscription lease, and journal the local removal.
    bool forget(region_id id) {
        if(!items_.contains(id))return false;
        record({++serial_,0,id,false});
        items_.erase(id);
        return true;
    }
    std::size_t resident_count() const noexcept {
        return resident_;
    }
    std::uint64_t serial() const noexcept {return serial_;}
    // Caller must resnapshot when the cursor has fallen behind the journal.
    std::vector<change> since(std::uint64_t cursor) const {
        if(cursor>serial_)throw std::invalid_argument("Future region cursor");
        if(!journal_.empty() && cursor<journal_.front().serial-1)
            throw std::out_of_range("Region journal expired; resnapshot required");
        std::vector<change> result;
        for(const auto& event:journal_)
            if(event.serial>cursor)result.push_back(event);
        return result;
    }
    template<class Visit> void each(Visit&& visit) const {
        for(const auto& [id,item]:items_)
            if(item.value)visit(id,item.revision,*item.value);
    }
    void clear() {
        items_.clear();previously_resident_.clear();
        journal_.clear();serial_=0;resident_=0;
    }
private:
    void record(change event) {
        const auto it=items_.find(event.id);
        if(event.present) {
            if(!previously_resident_.erase(event.id))++resident_;
        } else if(previously_resident_.erase(event.id)) {
            --resident_;
        }
        if(event.present)previously_resident_.emplace(event.id);
        journal_.push_back(event);
        if(journal_.size()>limit_)journal_.pop_front();
    }
    std::unordered_map<region_id,entry,region_hash> items_;
    std::unordered_set<region_id,region_hash> previously_resident_;
    std::deque<change> journal_;
    std::size_t resident_=0,limit_=8192;
    std::uint64_t serial_=0;
};
} // namespace vs::spatial
