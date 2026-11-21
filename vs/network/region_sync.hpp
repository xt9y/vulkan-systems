#pragma once
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
#include <vs/network/region.hpp>
#include <vs/spatial/regions.hpp>

namespace vs::network::region
{
// Reliable region replication policy. It does not open sockets, generate
// geometry or know the type of a region. Queue callbacks should use the
// existing authenticated network::session and return false on backpressure.
template <class ClientId>
class sender
{
    using id = spatial::region_id;
    using map_hash = spatial::region_hash;

    struct pending_release
    {
        std::uint32_t lease = 0;
        std::uint64_t revision = 1;
    };

    struct peer
    {
        std::unordered_map<id, std::uint32_t, map_hash> wanted;
        std::unordered_map<id, pending_release, map_hash> releases;
        std::unordered_map<id, std::uint64_t, map_hash> sent;
        std::uint32_t next_lease = 0;
    };

    static bool ordered(id a, id b)
    {
        if (a.y != b.y) return a.y < b.y;
        if (a.z != b.z) return a.z < b.z;

        return a.x < b.x;
    }

public:
    void subscribe(const ClientId &client, id region)
    {
        auto &peer = peers_[client];

        if (peer.wanted.contains(region)) return;
        peer.next_lease = (peer.next_lease + 1u) & 0xffffffu;

        if (!peer.next_lease) peer.next_lease = 1u;
        peer.wanted[region] = peer.next_lease;
        peer.releases.erase(region);
        peer.sent.erase(region);
    }

    void unsubscribe(const ClientId &client, id region)
    {
        const auto it = peers_.find(client);

        if (it == peers_.end()) return;
        auto &peer = it->second;

        const auto found = peer.wanted.find(region);

        if (found == peer.wanted.end()) return;

        const auto sent = peer.sent.find(region);
        // Release is ordered through the existing reliable channel; the
        // client discards its old local revision and GPU/CPU region payload.
        peer.releases[region] = {found->second, sent != peer.sent.end() ? sent->second : 1u};
        peer.sent.erase(region);
        peer.wanted.erase(found);
    }

    void disconnect(const ClientId &client) { peers_.erase(client); }
    void clear() { peers_.clear(); }
    std::size_t clients() const noexcept { return peers_.size(); }

    // A queue callback only returns true when the complete VSR1 packet has
    // been accepted by the reliable session's bounded outgoing queue.
    template <class Payload, class Encode, class Queue>
    std::size_t pump(const ClientId &client,
                     const spatial::region_store<Payload> &catalog,
                     Encode &&encode, Queue &&queue,
                     std::size_t message_budget = 4,
                     std::size_t byte_budget = 1024u * 1024u)
    {
        const auto found = peers_.find(client);
        if (found == peers_.end() || !message_budget) return 0;
        auto &peer = found->second;
        std::vector<id> unload;
        unload.reserve(peer.releases.size());

        for (const auto &[key, _] : peer.releases)
            unload.push_back(key);
        std::sort(unload.begin(), unload.end(), ordered);
        std::size_t count = 0;

        for (auto region : unload)
        {
            if (count >= message_budget || byte_budget < header_bytes) return count;

            const auto pending = peer.releases.at(region);

            message msg{};
            msg.action = operation::release;
            msg.id = region;
            msg.lease = pending.lease;
            msg.revision = pending.revision;

            if (!queue(::vs::network::region::encode(msg))) return count;
            peer.releases.erase(region);
            byte_budget -= header_bytes;
            ++count;
        }

        std::vector<id> wanted;
        wanted.reserve(peer.wanted.size());

        for (const auto &[key, _] : peer.wanted)
            wanted.push_back(key);
        std::sort(wanted.begin(), wanted.end(), ordered);

        for (auto region : wanted)
        {
            if (count >= message_budget || byte_budget < header_bytes) break;

            const auto *entry = catalog.get(region);
            if (!entry || !entry->revision) continue;

            const auto previous = peer.sent.find(region);

            if (previous != peer.sent.end() && previous->second >= entry->revision)
                continue;
            message msg{};
            msg.id = region;
            msg.lease = peer.wanted.at(region);
            msg.revision = entry->revision;

            if (entry->value) {
                msg.action = operation::replace;
                msg.payload = encode(*entry->value);
            }
            else {
                msg.action = operation::remove;
            }

            if (msg.payload.size() > max_payload ||
                msg.payload.size() + header_bytes > byte_budget)
                continue;

            if (!queue(::vs::network::region::encode(msg))) break;
            byte_budget -= msg.payload.size() + header_bytes;
            peer.sent[region] = entry->revision;
            ++count;
        }

        return count;
    }

private:
    std::unordered_map<ClientId, peer> peers_;
};

// Receiver keeps authoritative revisions and ignores duplicates, reordering
// and delayed replies to old interests. Payload contents are caller-defined.
class receiver
{
public:
    using payload_ptr = std::shared_ptr<const std::vector<std::uint8_t>>;

    struct update
    {
        ::vs::spatial::region_id id{};
        std::uint64_t revision = 0;

        payload_ptr payload; // Null on authoritative remove.
    };

    std::uint64_t serial() const noexcept { return regions_.serial(); }
    // The caller retains its own cursor. An expired journal throws, so
    // consumers can discard derived state and call snapshot() instead.
    std::vector<update> since(std::uint64_t cursor) const
    {
        const auto changes = regions_.since(cursor);

        std::unordered_set<::vs::spatial::region_id,
                           ::vs::spatial::region_hash>
            ids;
        std::vector<update> result;

        for (const auto &change : changes)
            ids.insert(change.id);
        result.reserve(ids.size());

        for (const auto id : ids)
        {
            const auto *item = regions_.get(id);

            if (item) result.push_back({id, item->revision, item->value});
            else
                result.push_back({id, 0, {}});
        }

        std::sort(result.begin(), result.end(), [](const auto &a, const auto &b)
                  {
            if(a.id.y!=b.id.y)return a.id.y<b.id.y;
            if(a.id.z!=b.id.z)return a.id.z<b.id.z;
            return a.id.x<b.id.x; });
        return result;
    }

    std::vector<update> snapshot() const
    {
        std::vector<update> result;
        regions_.each([&](auto id, auto revision, const auto &)
                      {
            const auto* entry=regions_.get(id);
            result.push_back({id,revision,entry->value}); });
        return result;
    }

    struct batch
    {
        std::uint64_t cursor = 0;

        bool full = false; // Consumer must discard old derived region state.
        std::vector<update> entries;
    };

    batch poll(std::uint64_t &cursor) const
    {
        batch result{};
        try
        {
            result.entries = since(cursor);
        }

        catch (const std::out_of_range &)
        {
            result.full = true;
            result.entries = snapshot();
        }

        catch (const std::invalid_argument &)
        {
            // Receiver reset or reconnected, so the old cursor belongs
            // to a different replication lifetime.
            result.full = true;
            result.entries = snapshot();
        }

        result.cursor = regions_.serial();
        cursor = result.cursor;

        return result;
    }

    bool apply(const message &m)
    {
        if (m.action != operation::replace && m.action != operation::remove &&
            m.action != operation::release) return false;
        auto it = leases_.find(m.id);

        if (it != leases_.end() && m.lease != it->second.value) {
            // 24-bit serial arithmetic rejects delayed packets from an
            // older subscription even after the region has been unloaded.
            const std::uint32_t gap = (m.lease - it->second.value) & 0xffffffu;

            if (!gap || gap >= 0x800000u) return false;
        }

        if (it == leases_.end() || m.lease != it->second.value) {
            regions_.forget(m.id);
            it = leases_.insert_or_assign(m.id, lease_state{m.lease, false}).first;
        }

        if (m.action == operation::release) {
            if (it->second.released) return false;
            it->second.released = true;
            regions_.forget(m.id);

            return true;
        }

        if (it->second.released) return false;
        if (m.action == operation::replace)
            return regions_.publish(m.id, m.revision,
                                    std::make_shared<const std::vector<std::uint8_t>>(m.payload));
        return regions_.remove(m.id, m.revision);
    }

    const spatial::region_store<std::vector<std::uint8_t>> &regions()
        const noexcept { return regions_; }
    void clear()
    {
        regions_.clear();
        leases_.clear();
    }

private:
    struct lease_state
    {
        std::uint32_t value = 0;

        bool released = false;
    };

    std::unordered_map<spatial::region_id, lease_state,
                       spatial::region_hash>
        leases_;
    spatial::region_store<std::vector<std::uint8_t>> regions_;
};
} // namespace vs::network::region
