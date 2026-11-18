#pragma once
#include <vs/network/region_sync.hpp>
#include <vs/scene/region_runtime.hpp>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace vs::scene {
// Decode and apply authoritative region fragments with an explicit budget.
// The network::region::receiver and runtime remain caller-owned. This adapter
// does not know how regions are generated, serialized or rendered.
class region_stream {
public:
    using update=::vs::network::region::receiver::update;
    explicit region_stream(std::size_t max_pending=8192)
        :max_pending_(std::max<std::size_t>(1,max_pending)) {}

    // Apply changes on the owning thread, after acquiring any network mutex.
    // decode(payload) returns shared_ptr<const region_fragment>.
    // publish updates only one region's broad-phase and invalidation masks.
    template<class Decode>
    std::size_t apply(const ::vs::network::region::receiver& receiver,
                      region_runtime& destination,Decode&& decode,
                      std::size_t budget=4) {
        auto batch=receiver.poll(cursor_);
        if(batch.full) {
            pending_.clear();
            destination.clear_and_invalidate();
        }
        for(auto& entry:batch.entries) {
            if(const auto found=pending_.find(entry.id);found!=pending_.end()) {
                found->second=std::move(entry);
                continue;
            }
            pending_.emplace(entry.id,std::move(entry));
        }
        // If the application falls behind the journal, resnapshot instead
        // of retaining unbounded intermediate revisions.
        if(pending_.size()>max_pending_) {
            pending_.clear();
            for(auto& entry:receiver.snapshot())
                pending_[entry.id]=std::move(entry);
            destination.clear_and_invalidate();
        }
        if(!budget || pending_.empty())return 0;
        std::vector<::vs::spatial::region_id> ids;
        ids.reserve(pending_.size());
        for(const auto& [id,_]:pending_)ids.push_back(id);
        std::sort(ids.begin(),ids.end(),[](auto a,auto b) {
            if(a.y!=b.y)return a.y<b.y;
            if(a.z!=b.z)return a.z<b.z;
            return a.x<b.x;
        });
        std::size_t applied=0;
        for(auto id:ids) {
            if(applied>=budget)break;
            const auto it=pending_.find(id);
            if(it==pending_.end())continue;
            const auto& entry=it->second;
            if(entry.payload) {
                auto fragment=decode(*entry.payload);
                if(!fragment)
                    throw std::runtime_error("Region decoder returned null");
                destination.publish(id,entry.revision,std::move(fragment));
            } else if(entry.revision) {
                destination.remove(id,entry.revision);
            } else {
                destination.release(id);
            }
            pending_.erase(it);
            ++applied;
        }
        return applied;
    }
    void reset(){pending_.clear();cursor_=0;}
    std::size_t pending() const noexcept{return pending_.size();}
    std::uint64_t cursor() const noexcept{return cursor_;}
private:
    std::uint64_t cursor_=0;
    std::size_t max_pending_=8192;
    std::unordered_map<::vs::spatial::region_id,update,
                       ::vs::spatial::region_hash> pending_;
};
} // namespace vs::scene
