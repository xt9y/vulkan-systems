#pragma once

#include "endpoint.hpp"
#include "socket.hpp"
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace net::secure {
// One transport: DTLS 1.2 with automatic persistent server identity and
// client-side trust-on-first-use public-key pinning. No plaintext fallback.
enum class role { client, server };
struct settings {
    // A persistent per-user directory, never inside the repository.
    std::string storage_directory;
    // Client: configured DNS/IP and port. Trust pins are scoped per endpoint.
    std::string trust_identity;
    std::size_t max_peers = 64;
    std::size_t max_delivered_bytes = 4 * 1024 * 1024;
    std::size_t max_pending_bytes_per_peer = 64 * 1024;
};
struct state;
using resource = std::shared_ptr<state>;
resource create(role side, settings config, std::uint16_t port = 0,
                const std::string& bind_host = "0.0.0.0");
void destroy(resource& connection) noexcept;
// Called initially and on a reconnect: discards the previous DTLS session,
// regenerates the handshake, and retains the pinned server key on disk.
void connect(const resource& connection, const endpoint::resource& remote);
bool send(const resource& connection, const endpoint::resource& peer,
          const void* bytes, std::size_t size);
std::optional<std::size_t> receive(const resource& connection,
                                   endpoint::resource& peer, void* bytes,
                                   std::size_t capacity);
void wait_readable(const resource& connection,
                   std::chrono::milliseconds timeout);
bool authenticated(const resource& connection, const endpoint::resource& peer);
void disconnect(const resource& connection, const endpoint::resource& peer);
} // namespace net::secure
