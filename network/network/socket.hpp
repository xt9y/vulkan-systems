#pragma once

#include "endpoint.hpp"

#include <chrono>
#include <cstddef>
#include <optional>

namespace net::socket
{
    struct state
    {
        bool initialized = false;
    };

    struct resource
    {
#ifdef _WIN32
        SOCKET handle = INVALID_SOCKET;
#else
        int handle = -1;
#endif
    };

    state* get();
    void init();
    void shutdown() noexcept;

    resource create(std::uint16_t port = 0, const std::string& bind_host = "0.0.0.0");
    void destroy(resource& socket) noexcept;

    void wait_readable(const resource& socket, std::chrono::milliseconds timeout);
    bool send(
        const resource& socket, const endpoint::resource& peer, const void* data, std::size_t size
    );
    std::optional<std::size_t>
    receive(const resource& socket, endpoint::resource& peer, void* data, std::size_t capacity);
} // namespace net::socket
