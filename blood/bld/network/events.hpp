#pragma once
#include "../core/types.hpp"
#include <array>
#include <deque>
#include <optional>
#include <unordered_set>

namespace bld::network
{
    // No sockets or networking framework. Send these bytes with any transport.
    using packet = std::array<std::uint8_t, 84>;

    struct state
    {
        std::deque<std::uint64_t> history;
        std::unordered_set<std::uint64_t> seen;
        std::size_t replay_window = 1024;
    };

    state *get();
    void init(std::size_t replay_window = 1024);
    packet encode(const impact &event);
    std::optional<impact> decode(const void *bytes, std::size_t count);
    bool receive(const void *bytes, std::size_t count);
    void reset();
    void shutdown();
}
