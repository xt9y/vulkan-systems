#pragma once
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vs/jobs/pool.hpp>
#include <vs/spatial/regions.hpp>

namespace vs::spatial
{
// Reusable asynchronous producer for spatial region payloads. The generator
// has no Vulkan, network or game dependency. All request/commit/remove calls
// must occur on the owning thread; only generator() runs on a worker.
// A cancelled/replaced request cannot publish after it completes.
template <class Payload>
class region_scheduler
{
public:
    using payload_ptr = std::shared_ptr<const Payload>;
    using generator = std::function<payload_ptr(region_id)>;

    explicit region_scheduler(::vs::jobs::pool &workers, generator generate,
                              std::size_t journal_limit = 8192)
        : workers_(workers), generate_(std::move(generate)),
          state_(std::make_shared<state>(journal_limit))
    {
        if (!generate_) throw std::invalid_argument("Region generator is empty");
    }

    region_scheduler(const region_scheduler &) = delete;
    region_scheduler &operator=(const region_scheduler &) = delete;

    // A revision identifies the requested content, not a queue position.
    // Repeating a request at the same revision is a no-op.
    bool request(region_id id, std::uint64_t revision)
    {
        if (!revision) throw std::invalid_argument("Region revision is zero");
        auto &active = state_->active;

        if (const auto it = active.find(id); it != active.end() &&
                                             it->second.revision >= revision) return false;
        if (const auto *item = state_->regions.get(id);
            item && item->revision >= revision) return false;
        const auto token = state_->next_token + 1u;

        std::weak_ptr<state> weak = state_;

        const auto generator_copy = generate_;

        if (!workers_.submit([weak, generator_copy, id, revision, token]()
                                 -> ::vs::jobs::completion
                             {
            auto result=generator_copy(id);
            if(!result)throw std::runtime_error("Region producer returned null");
            return [weak,id,revision,token,result=std::move(result)]() mutable {
                auto shared=weak.lock();
                if(!shared)return; // Scheduler destroyed or reset.
                const auto it=shared->active.find(id);
                if(it==shared->active.end() || it->second.token!=token)return;
                shared->active.erase(it);
                shared->regions.publish(id,revision,std::move(result));
            }; })) return false;
        state_->next_token = token;
        active[id] = {revision, token};

        return true;
    }

    // Tombstones and request tokens reject late worker completions.
    bool remove(region_id id, std::uint64_t revision)
    {
        if (!revision) throw std::invalid_argument("Region revision is zero");
        if (const auto it = state_->active.find(id);
            it != state_->active.end() && revision < it->second.revision) return false;
        if (const auto *current = state_->regions.get(id);
            current && revision <= current->revision) return false;
        state_->active.erase(id);

        return state_->regions.remove(id, revision);
    }
    // jobs::pool::commit executes completion callbacks on the owning thread.
    // Pool-level failures are propagated; a failed job never becomes resident.
    std::size_t commit(std::size_t budget = 16) { return workers_.commit(budget); }
    bool pending(region_id id) const { return state_->active.contains(id); }
    std::size_t pending() const noexcept { return state_->active.size(); }
    const region_store<Payload> &regions() const noexcept { return state_->regions; }
    // In-flight tasks retain only weak ownership; safe to reset while busy.
    void clear() { state_ = std::make_shared<state>(state_->journal_limit); }

private:
    struct request_state
    { std::uint64_t revision = 0, token = 0; };
    struct state
    {
        explicit state(std::size_t limit) : journal_limit(limit), regions(limit) {}
        std::size_t journal_limit;
        region_store<Payload> regions;
        std::unordered_map<region_id, request_state, region_hash> active;
        std::uint64_t next_token = 0;
    };
    ::vs::jobs::pool &workers_;

    generator generate_;

    std::shared_ptr<state> state_;
};
} // namespace vs::spatial
