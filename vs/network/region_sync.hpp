#pragma once
#include <vs/network/region.hpp>
#include <vs/spatial/regions.hpp>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace vs::network::region {
// Reliable region replication policy. It does not open sockets, generate
// geometry or know the type of a region. Queue callbacks should use the
// existing authenticated network::session and return false on backpressure.
template<class ClientId> class sender {
    struct peer {
        std::unordered_set<spatial::region_id,spatial::region_hash> wanted;
        std::unordered_map<spatial::region_id,std::uint64_t,
                           spatial::region_hash> sent;
    };
public:
    void subscribe(const ClientId& client,spatial::region_id id) {
        peers_[client].wanted.insert(id);
    }
    void unsubscribe(const ClientId& client,spatial::region_id id) {
        if(const auto it=peers_.find(client);it!=peers_.end()) {
            it->second.wanted.erase(id);
            it->second.sent.erase(id);
        }
    }
    void disconnect(const ClientId& client) {peers_.erase(client);}
    void clear() {peers_.clear();}
    std::size_t clients() const noexcept {return peers_.size();}

    // Wire format is stable regardless of Payload or generator implementation.
    // The encode callback converts a payload to compressed region bytes.
    // The queue callback accepts a complete message into network::session.
    template<class Payload,class Encode,class Queue>
    std::size_t pump(const ClientId& client,
                     const spatial::region_store<Payload>& catalog,
                     Encode&& encode,Queue&& queue,
                     std::size_t message_budget=4,
                     std::size_t byte_budget=1024u*1024u) {
        const auto found=peers_.find(client);
        if(found==peers_.end() || !message_budget)return 0;
        auto& peer=found->second;
        std::vector<spatial::region_id> ids(peer.wanted.begin(),peer.wanted.end());
        std::sort(ids.begin(),ids.end(),[](auto a,auto b) {
            if(a.y!=b.y)return a.y<b.y;
            if(a.z!=b.z)return a.z<b.z;
            return a.x<b.x;
        });
        std::size_t sent=0;
        for(auto id:ids) {
            if(sent>=message_budget || byte_budget<header_bytes)break;
            const auto* entry=catalog.get(id);
            if(!entry || !entry->revision)continue;
            const auto previous=peer.sent.find(id);
            if(previous!=peer.sent.end() && previous->second>=entry->revision)
                continue;
            message next{};
            next.id=id;
            next.revision=entry->revision;
            if(entry->value) {
                next.action=operation::replace;
                next.payload=encode(*entry->value);
            } else {
                next.action=operation::remove;
            }
            if(next.payload.size()>max_payload ||
               next.payload.size()+header_bytes>byte_budget)
                continue;
            auto bytes=::vs::network::region::encode(next);
            if(!queue(std::move(bytes)))break;
            byte_budget-=next.payload.size()+header_bytes;
            peer.sent[id]=entry->revision;
            ++sent;
        }
        return sent;
    }
private:
    std::unordered_map<ClientId,peer> peers_;
};

// Receiver keeps authoritative revisions and ignores duplicates, reordering
// and delayed replies to old interests. Payload contents are caller-defined.
class receiver {
public:
    bool apply(const message& m) {
        if(m.action!=operation::replace && m.action!=operation::remove)
            return false;
        if(m.action==operation::replace)
            return regions_.publish(m.id,m.revision,
                std::make_shared<const std::vector<std::uint8_t>>(m.payload));
        return regions_.remove(m.id,m.revision);
    }
    const spatial::region_store<std::vector<std::uint8_t>>& regions()
        const noexcept {return regions_;}
    void clear(){regions_.clear();}
private:
    spatial::region_store<std::vector<std::uint8_t>> regions_;
};
} // namespace vs::network::region
