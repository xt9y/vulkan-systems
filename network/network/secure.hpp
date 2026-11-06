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
// DTLS 1.2 mutually authenticated by X.509 certificates. No plaintext fallback.
// Credentials belong to the host application, never inside VulkanSystems.
enum class role { client, server };
struct settings {
    std::string ca_file;
    std::string certificate_file;
    std::string private_key_file;
    // Client: expected server DNS/IP identity used for certificate verification.
    std::string server_name;
    std::size_t max_peers = 64;
    std::size_t max_delivered_bytes = 4 * 1024 * 1024;
    std::size_t max_pending_bytes_per_peer = 64 * 1024;
};
struct state;
using resource = std::shared_ptr<state>;
resource create(role side, settings config, std::uint16_t port = 0,
                const std::string& bind_host = "0.0.0.0");
void destroy(resource& connection) noexcept;
// Required for clients: pin the UDP endpoint before accepting DTLS traffic.
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
