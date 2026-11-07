#include "secure.hpp"
#include "identity.hpp"

#include <mbedtls/ssl.h>
#include <mbedtls/ssl_ciphersuites.h>
#include <mbedtls/ssl_cookie.h>
#include <mbedtls/x509_crt.h>
#include <mbedtls/pk.h>
#include <mbedtls/entropy.h>
#include <mbedtls/ctr_drbg.h>
#include <mbedtls/timing.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <deque>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace net::secure {
namespace {
using clock = std::chrono::steady_clock;
using bytes = std::vector<unsigned char>;
constexpr std::size_t max_ciphertext = 2048;
constexpr std::size_t max_plaintext = 1400;
constexpr std::size_t max_datagrams_per_poll = 256;
constexpr auto unauthenticated_timeout = std::chrono::seconds(10);
constexpr auto authenticated_timeout = std::chrono::seconds(45);

struct peer_state;
struct incoming_message {
    endpoint::resource peer;
    bytes payload;
};

static void require(int code, const char* label) {
    if(code != 0) throw std::runtime_error(label);
}

} // namespace

struct state {
    role side;
    settings limits;
    socket::resource udp{};
    mbedtls_entropy_context entropy{};
    mbedtls_ctr_drbg_context drbg{};
    mbedtls_x509_crt ca{};
    mbedtls_x509_crt identity{};
    mbedtls_pk_context private_key{};
    mbedtls_ssl_config tls_config{};
    mbedtls_ssl_cookie_ctx cookies{};
    std::map<std::string, std::unique_ptr<peer_state>> peers;
    std::deque<incoming_message> delivered;
    std::size_t delivered_bytes = 0;
    std::string fixed_remote;

    state(role side_, settings options) : side(side_), limits(std::move(options)) {
        mbedtls_entropy_init(&entropy);
        mbedtls_ctr_drbg_init(&drbg);
        mbedtls_x509_crt_init(&ca);
        mbedtls_x509_crt_init(&identity);
        mbedtls_pk_init(&private_key);
        mbedtls_ssl_config_init(&tls_config);
        mbedtls_ssl_cookie_init(&cookies);
    }
    ~state();
};

namespace {
struct peer_state {
    state* owner;
    endpoint::resource endpoint;
    mbedtls_ssl_context tls{};
    mbedtls_timing_delay_context timer{};
    std::deque<bytes> inbound;
    std::deque<bytes> outgoing;
    std::size_t outgoing_bytes = 0;
    clock::time_point last_seen = clock::now();
    bool authorized = false;

    peer_state(state* s, const endpoint::resource& p) : owner(s), endpoint(p) {
        mbedtls_ssl_init(&tls);
    }
    ~peer_state() { mbedtls_ssl_free(&tls); }
};

int transmit(void* ctx, const unsigned char* packet, std::size_t count) {
    auto& p = *static_cast<peer_state*>(ctx);
    if(count == 0 || count > max_ciphertext)
        return MBEDTLS_ERR_SSL_INTERNAL_ERROR;
    try {
        if(!socket::send(p.owner->udp, p.endpoint, packet, count))
            return MBEDTLS_ERR_SSL_WANT_WRITE;
        return static_cast<int>(count);
    } catch(...) {
        return MBEDTLS_ERR_SSL_INTERNAL_ERROR;
    }
}

int retrieve(void* ctx, unsigned char* destination, std::size_t capacity) {
    auto& p = *static_cast<peer_state*>(ctx);
    if(p.inbound.empty()) return MBEDTLS_ERR_SSL_WANT_READ;
    auto& packet = p.inbound.front();
    if(packet.size() > capacity)
        return MBEDTLS_ERR_SSL_BUFFER_TOO_SMALL;
    std::memcpy(destination, packet.data(), packet.size());
    const auto count = static_cast<int>(packet.size());
    p.inbound.pop_front();
    return count;
}

void configure_peer(peer_state& p) {
    require(mbedtls_ssl_setup(&p.tls, &p.owner->tls_config),
            "DTLS peer setup failed");
    mbedtls_ssl_set_bio(&p.tls, &p, transmit, retrieve, nullptr);
    mbedtls_ssl_set_timer_cb(&p.tls, &p.timer,
                             mbedtls_timing_set_delay,
                             mbedtls_timing_get_delay);
    mbedtls_ssl_set_mtu(&p.tls, 1400);
    if(p.owner->side == role::client && !p.owner->limits.automatic_identity) {
        require(mbedtls_ssl_set_hostname(
            &p.tls, p.owner->limits.server_name.c_str()),
            "DTLS server identity setup failed");
    } else if(p.owner->side == role::server) {
        const auto id = endpoint::identifier(p.endpoint);
        require(mbedtls_ssl_set_client_transport_id(
            &p.tls, reinterpret_cast<const unsigned char*>(id.data()),
            id.size()), "DTLS cookie peer identity failed");
    }
}

peer_state* ensure(state& s, const endpoint::resource& address) {
    const auto id = endpoint::identifier(address);
    if(id.empty()) return nullptr;
    if(s.side == role::client &&
       (s.fixed_remote.empty() || s.fixed_remote != id))
        return nullptr;
    auto it = s.peers.find(id);
    if(it != s.peers.end()) return it->second.get();
    if(s.peers.size() >= s.limits.max_peers) return nullptr;
    auto p = std::make_unique<peer_state>(&s, address);
    configure_peer(*p);
    auto* result = p.get();
    s.peers.emplace(id, std::move(p));
    return result;
}

bool advance(peer_state& p) {
    if(p.authorized) return true;
    const int result = mbedtls_ssl_handshake(&p.tls);
    if(result == 0) {
        if(p.owner->limits.automatic_identity) {
            if(p.owner->side == role::client) {
                const auto* certificate=mbedtls_ssl_get_peer_cert(&p.tls);
                if(!certificate || !identity::verify_or_pin(
                    p.owner->limits.storage_directory,
                    p.owner->limits.trust_identity,*certificate))
                    return false;
            }
        } else if(mbedtls_ssl_get_verify_result(&p.tls) != 0) {
            return false;
        }
        p.authorized = true;
        return true;
    }
    if(result == MBEDTLS_ERR_SSL_HELLO_VERIFY_REQUIRED &&
       p.owner->side == role::server) {
        if(mbedtls_ssl_session_reset(&p.tls) != 0) return false;
        const auto id = endpoint::identifier(p.endpoint);
        if(mbedtls_ssl_set_client_transport_id(
            &p.tls, reinterpret_cast<const unsigned char*>(id.data()),
            id.size()) != 0) return false;
        return true;
    }
    return result == MBEDTLS_ERR_SSL_WANT_READ ||
           result == MBEDTLS_ERR_SSL_WANT_WRITE;
}

bool flush(peer_state& p) {
    if(!p.authorized) return true;
    while(!p.outgoing.empty()) {
        auto& packet = p.outgoing.front();
        const int sent = mbedtls_ssl_write(&p.tls, packet.data(),
                                            packet.size());
        if(sent == MBEDTLS_ERR_SSL_WANT_READ ||
           sent == MBEDTLS_ERR_SSL_WANT_WRITE) return true;
        if(sent < 0 || static_cast<std::size_t>(sent) != packet.size())
            return false;
        p.outgoing_bytes -= packet.size();
        p.outgoing.pop_front();
    }
    return true;
}

bool read_plaintext(peer_state& p) {
    if(!p.authorized) return true;
    auto& s = *p.owner;
    std::array<unsigned char, max_plaintext + 1> buffer{};
    for(int i = 0; i < 128; ++i) {
        const int count = mbedtls_ssl_read(&p.tls, buffer.data(),
                                           buffer.size());
        if(count == MBEDTLS_ERR_SSL_WANT_READ ||
           count == MBEDTLS_ERR_SSL_WANT_WRITE) return true;
        if(count <= 0) return false;
        if(s.delivered_bytes + std::size_t(count) >
           s.limits.max_delivered_bytes) continue;
        s.delivered.push_back({p.endpoint,
            bytes(buffer.begin(), buffer.begin() + count)});
        s.delivered_bytes += std::size_t(count);
    }
    return true;
}

bool drive(peer_state& p) {
    if(!advance(p)) return false;
    if(!flush(p)) return false;
    return read_plaintext(p);
}

void expire(state& s) {
    const auto now = clock::now();
    for(auto it = s.peers.begin(); it != s.peers.end();) {
        const auto timeout = it->second->authorized ?
            authenticated_timeout : unauthenticated_timeout;
        if(now - it->second->last_seen > timeout)
            it = s.peers.erase(it);
        else {
            // Drive DTLS retransmission timers even without incoming packets.
            if(!drive(*it->second))
                it = s.peers.erase(it);
            else ++it;
        }
    }
}

void process_datagram(state& s, const endpoint::resource& from,
                      const unsigned char* bytes, std::size_t size) {
    // Never allocate client state for a forged remote endpoint.
    if(s.side == role::client &&
       endpoint::identifier(from) != s.fixed_remote) return;
    // Reject traffic without a valid DTLS record header before allocation.
    if(size < 13 || bytes[0] < 20 || bytes[0] > 25 ||
       bytes[1] != 0xfe) return;
    // Authentication is mandatory; no application packet bypasses TLS.
    auto* p = ensure(s, from);
    if(!p) return;
    p->last_seen = clock::now();
    if(p->inbound.size() >= 32) return;
    p->inbound.emplace_back(bytes, bytes + size);
    if(!drive(*p)) s.peers.erase(endpoint::identifier(from));
}
} // namespace

state::~state() {
    peers.clear();
    socket::destroy(udp);
    mbedtls_ssl_cookie_free(&cookies);
    mbedtls_ssl_config_free(&tls_config);
    mbedtls_pk_free(&private_key);
    mbedtls_x509_crt_free(&identity);
    mbedtls_x509_crt_free(&ca);
    mbedtls_ctr_drbg_free(&drbg);
    mbedtls_entropy_free(&entropy);
}

resource create(role side, settings config, std::uint16_t port,
                const std::string& bind_host) {
    if(config.automatic_identity) {
        if(config.storage_directory.empty() ||
           (side == role::client && config.trust_identity.empty()))
            throw std::invalid_argument("Automatic DTLS requires persistent storage and a stable server identity");
        if(!config.ca_file.empty() || !config.certificate_file.empty() ||
           !config.private_key_file.empty())
            throw std::invalid_argument("Automatic DTLS and explicit certificates cannot be mixed");
    } else if(config.ca_file.empty() || config.certificate_file.empty() ||
              config.private_key_file.empty() ||
              (side == role::client && config.server_name.empty())) {
        throw std::invalid_argument("Mutual TLS requires CA, local certificate, "
                                    "private key and verified server name");
    }
    if(!config.max_peers || !config.max_delivered_bytes ||
       config.max_pending_bytes_per_peer < max_plaintext)
        throw std::invalid_argument("invalid DTLS memory limits");
    auto s = std::make_shared<state>(side, std::move(config));
    constexpr unsigned char personalization[] = "VulkanNetwork DTLS";
    require(mbedtls_ctr_drbg_seed(&s->drbg, mbedtls_entropy_func,
             &s->entropy, personalization, sizeof(personalization) - 1),
             "DTLS random generator setup failed");
    if(s->limits.automatic_identity) {
        if(side == role::server)
            identity::local_server(s->limits.storage_directory,
                                   s->drbg,s->identity,s->private_key);
    } else {
        require(mbedtls_x509_crt_parse_file(
            &s->ca, s->limits.ca_file.c_str()), "DTLS CA cannot be loaded");
        require(mbedtls_x509_crt_parse_file(
            &s->identity, s->limits.certificate_file.c_str()),
            "DTLS local certificate cannot be loaded");
        require(mbedtls_pk_parse_keyfile(
            &s->private_key, s->limits.private_key_file.c_str(), nullptr,
            mbedtls_ctr_drbg_random, &s->drbg),
            "DTLS private key cannot be loaded");
    }
    require(mbedtls_ssl_config_defaults(
        &s->tls_config, side == role::server ?
        MBEDTLS_SSL_IS_SERVER : MBEDTLS_SSL_IS_CLIENT,
        MBEDTLS_SSL_TRANSPORT_DATAGRAM, MBEDTLS_SSL_PRESET_DEFAULT),
        "DTLS settings initialization failed");
    // Require forward-secret ECDHE and AEAD; reject RSA key transport/CBC.
    static const int allowed_suites[] = {
        MBEDTLS_TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256,
        MBEDTLS_TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256,
        0
    };
    mbedtls_ssl_conf_ciphersuites(&s->tls_config, allowed_suites);
    mbedtls_ssl_conf_rng(&s->tls_config, mbedtls_ctr_drbg_random, &s->drbg);
    mbedtls_ssl_conf_authmode(&s->tls_config,
        s->limits.automatic_identity ?
            (side == role::client ? MBEDTLS_SSL_VERIFY_OPTIONAL : MBEDTLS_SSL_VERIFY_NONE)
            : MBEDTLS_SSL_VERIFY_REQUIRED);
    if(!s->limits.automatic_identity)
        mbedtls_ssl_conf_ca_chain(&s->tls_config, &s->ca, nullptr);
    if(side == role::server || !s->limits.automatic_identity)
        require(mbedtls_ssl_conf_own_cert(
            &s->tls_config, &s->identity, &s->private_key),
            "DTLS certificate/key mismatch");
    mbedtls_ssl_conf_handshake_timeout(&s->tls_config, 1000, 8000);
    if(side == role::server) {
        require(mbedtls_ssl_cookie_setup(&s->cookies,
                mbedtls_ctr_drbg_random, &s->drbg),
                "DTLS cookie initialization failed");
        mbedtls_ssl_conf_dtls_cookies(&s->tls_config,
            mbedtls_ssl_cookie_write, mbedtls_ssl_cookie_check,
            &s->cookies);
    }
    socket::init();
    s->udp = socket::create(port, bind_host);
    return s;
}

void destroy(resource& r) noexcept { r.reset(); }

void connect(const resource& r, const endpoint::resource& remote) {
    if(!r || r->side != role::client)
        throw std::invalid_argument("DTLS connect requires a client socket");
    const auto id=endpoint::identifier(remote);
    if(id.empty())throw std::invalid_argument("invalid DTLS server endpoint");
    if(!r->fixed_remote.empty() && r->fixed_remote != id)
        throw std::logic_error("DTLS client endpoint already pinned");
    r->fixed_remote=id;
    auto* peer=ensure(*r,remote);
    if(!peer || !drive(*peer))
        throw std::runtime_error("DTLS client handshake could not start");
}

bool send(const resource& r, const endpoint::resource& remote,
          const void* data, std::size_t count) {
    if(!r || !data || !count || count > max_plaintext) return false;
    auto* p = ensure(*r, remote);
    if(!p) return false;
    if(count > r->limits.max_pending_bytes_per_peer -
       std::min(p->outgoing_bytes, r->limits.max_pending_bytes_per_peer))
        return false;
    auto* first = static_cast<const unsigned char*>(data);
    p->outgoing.emplace_back(first, first + count);
    p->outgoing_bytes += count;
    if(!drive(*p)) {
        r->peers.erase(endpoint::identifier(remote));
        return false;
    }
    return true;
}

std::optional<std::size_t> receive(const resource& r,
                                   endpoint::resource& sender,
                                   void* destination, std::size_t capacity) {
    if(!r || !destination) return std::nullopt;
    std::array<unsigned char, max_ciphertext + 1> packet{};
    // Progress handshake/DTLS traffic without confusing "no plaintext yet"
    // with an empty UDP socket, which would otherwise stall reassembly.
    for(std::size_t i = 0; i < max_datagrams_per_poll; ++i) {
        if(!r->delivered.empty()) break;
        endpoint::resource remote;
        const auto size = socket::receive(
            r->udp, remote, packet.data(), packet.size());
        if(!size) break;
        if(*size > max_ciphertext) continue;
        process_datagram(*r, remote, packet.data(), *size);
    }
    expire(*r);
    if(r->delivered.empty()) return std::nullopt;
    auto& item = r->delivered.front();
    if(item.payload.size() > capacity)
        throw std::out_of_range("DTLS plaintext receive buffer too small");
    std::memcpy(destination, item.payload.data(), item.payload.size());
    sender = item.peer;
    const auto size = item.payload.size();
    r->delivered_bytes -= size;
    r->delivered.pop_front();
    return size;
}

void wait_readable(const resource& r, std::chrono::milliseconds timeout) {
    if(r) socket::wait_readable(r->udp, timeout);
}

bool authenticated(const resource& r, const endpoint::resource& remote) {
    if(!r) return false;
    const auto it = r->peers.find(endpoint::identifier(remote));
    return it != r->peers.end() && it->second->authorized;
}

void disconnect(const resource& r, const endpoint::resource& remote) {
    if(r) r->peers.erase(endpoint::identifier(remote));
}
} // namespace net::secure
