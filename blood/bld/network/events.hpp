#pragma once
#include "../core/types.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>
#include <unordered_set>

namespace bld::network
{
    // Network-transport agnostic: no socket, connection or server ownership.
    using packet=std::array<std::uint8_t,88>;

    struct state
    {
        std::mutex mutex;
        std::deque<impact> queue;
        std::deque<std::uint64_t> history;
        std::unordered_set<std::uint64_t> seen;
        std::size_t window=2048;
    };

    state *get();
    void init(std::size_t replay_window=2048);
    packet send(const impact &event);
    std::optional<impact> decode(const void *bytes,std::size_t size);
    bool receive(const void *bytes,std::size_t size);
    // Called from bld::system::update, NOT the network receive thread.
    void dispatch();
    void reset();
    void shutdown();
}
