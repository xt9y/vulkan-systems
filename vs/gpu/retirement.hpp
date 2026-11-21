#pragma once
#include <algorithm>
#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

namespace vs::gpu
{
// Resources are destroyed only after the frame scheduler reports a completed
// submission serial. Never call collect with a guessed/current frame index.
class retirement
{
public:
    void after(std::uint64_t last_use, std::function<void()> cleanup)
    {
        if (cleanup) pending_.push_back({last_use, std::move(cleanup)});
    }

    std::size_t collect(std::uint64_t completed_serial)
    {
        std::size_t count = 0;

        for (std::size_t i = 0; i < pending_.size();)
        {
            if (pending_[i].last_use > completed_serial) {
                ++i;

                continue;
            }

            auto cleanup = std::move(pending_[i].cleanup);

            pending_.erase(pending_.begin() + i);
            cleanup();
            ++count;
        }

        return count;
    }

    std::size_t pending() const noexcept { return pending_.size(); }
    // Call only after vk::device::idle() or device teardown is synchronized.
    void drain_idle()
    {
        for (auto &item : pending_)
            item.cleanup();
        pending_.clear();
    }

private:
    struct entry
    {
        std::uint64_t last_use;
        std::function<void()> cleanup;
    };

    std::vector<entry> pending_;
};
} // namespace vs::gpu
