#pragma once
#include "endpoint.hpp"
#include "socket.hpp"
#include "stream.hpp"
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>

namespace net {
    struct delivery { endpoint::resource peer; stream::bytes payload; };

    // Client/server-neutral datagram multiplexer: one reliable channel per peer.
    class host {
    public:
        explicit host(stream::settings limits = {});
        ~host() { stop(); }
        host(const host&)=delete;
        host& operator=(const host&)=delete;
        void start(std::uint16_t port=0,const std::string& bind="0.0.0.0");
        void stop() noexcept;
        bool send(const endpoint::resource& peer,stream::bytes payload);
        void pump(std::size_t max_receives=256,std::size_t byte_budget=256*1024);
        bool pop(delivery& received);
    private:
        struct client {
            endpoint::resource peer;
            stream::channel channel;
            stream::clock::time_point last_seen{};
            client(endpoint::resource peer,stream::settings limits)
                :peer(peer),channel(limits) {}
        };
        static std::uint64_t key(const endpoint::resource& address);
        stream::settings limits_;
        socket::resource socket_{};
        bool active_=false;
        std::map<std::uint64_t,client> peers_;
        std::deque<delivery> received_;
        std::size_t received_bytes_=0;
    };
}
