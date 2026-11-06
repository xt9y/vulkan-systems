#pragma once
#include "endpoint.hpp"
#include "socket.hpp"
#include "stream.hpp"
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <string>

namespace net {
// Reliable datagram session multiplexed over a caller-owned UDP socket.
// Unlike host, session can share the socket with a game's control packets.
struct delivery {
    endpoint::resource peer;
    stream::bytes payload;
};
class session {
public:
    struct settings {
        stream::settings channel{};
        std::size_t max_peers=64;
        std::size_t max_delivered_bytes=128*1024*1024;
        std::chrono::seconds idle_timeout{60};
        bool allow_unknown_peers=true;
    };
    session();
    explicit session(settings options);
    // Allow a previously authenticated peer when allow_unknown_peers is false.
    bool open(const endpoint::resource& peer);
    void close(const endpoint::resource& peer);
    // Send one complete message, fragmented by the channel.
    bool send(const endpoint::resource& peer,stream::bytes payload);
    // Returns true for VSN1 datagrams even if rejected as invalid.
    bool accept(const endpoint::resource& peer,const void* data,
                std::size_t length,stream::clock::time_point now=stream::clock::now());
    // Accepts any datagram sender (raw UDP or authenticated DTLS).
    using datagram_sender=std::function<bool(const endpoint::resource&,const void*,std::size_t)>;
    void pump(const datagram_sender& transmit,std::size_t byte_budget=256*1024,
              stream::clock::time_point now=stream::clock::now());
    // Compatibility adapter for dedicated UDP hosts.
    void pump(const socket::resource& udp,std::size_t byte_budget=256*1024,
              stream::clock::time_point now=stream::clock::now());
    bool pop(delivery& out);
    std::size_t pending(const endpoint::resource& peer) const;
    void clear() noexcept;
private:
    struct client {
        endpoint::resource peer;
        stream::channel channel;
        stream::clock::time_point last_seen{};
        client(const endpoint::resource& p,stream::settings limits)
            :peer(p),channel(limits),last_seen(stream::clock::now()){}
    };
    using clients=std::map<std::string,client>;
    clients::iterator find_or_open(const endpoint::resource& peer,bool unknown);
    settings options_;
    clients peers_;
    std::deque<delivery> received_;
    std::size_t received_bytes_=0;
};
}
