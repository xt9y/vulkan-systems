#include "host.hpp"
#include <array>
#include <algorithm>
#include <stdexcept>
#include <utility>
#include <tuple>

namespace net {
    host::host(stream::settings limits):limits_(limits) {}
    std::uint64_t host::key(const endpoint::resource& peer) {
        return (std::uint64_t(peer.address.sin_addr.s_addr)<<16)|
            std::uint64_t(peer.address.sin_port);
    }
    void host::start(std::uint16_t port,const std::string& bind) {
        if(active_)throw std::logic_error("network host already started");
        socket::init();
        socket_=socket::create(port,bind);
        active_=true;
    }
    void host::stop() noexcept {
        if(active_)socket::destroy(socket_);
        active_=false; peers_.clear(); received_.clear(); received_bytes_=0;
    }
    bool host::send(const endpoint::resource& peer,stream::bytes payload) {
        if(!active_)return false;
        const auto id=key(peer);
        auto it=peers_.find(id);
        const bool created=it==peers_.end();
        if(created) {
            if(peers_.size()>=256)return false;
            it=peers_.emplace(
                std::piecewise_construct,
                std::forward_as_tuple(id),
                std::forward_as_tuple(peer,limits_)).first;
        }
        it->second.last_seen=stream::clock::now();
        const bool accepted=it->second.channel.queue(std::move(payload))!=0;
        if(!accepted && created)
            peers_.erase(it);
        return accepted;
    }
    void host::pump(std::size_t receives,std::size_t byte_budget) {
        if(!active_)throw std::logic_error("network host is not started");
        std::array<std::uint8_t,1473> packet{};
        const auto now=stream::clock::now();
        for(std::size_t i=0;i<receives;++i) {
            endpoint::resource sender;
            const auto size=socket::receive(socket_,sender,packet.data(),packet.size());
            if(!size)break;
            if(*size>1472 || *size<28 || packet[0]!='V'||packet[1]!='S'||
               packet[2]!='N'||packet[3]!='1')continue;
            const auto id=key(sender);
            auto it=peers_.find(id);
            if(it==peers_.end()) {
                if(peers_.size()>=256)continue;
                it=peers_.emplace(std::piecewise_construct,
                    std::forward_as_tuple(id),std::forward_as_tuple(sender,limits_)).first;
            }
            if(!it->second.channel.accept(packet.data(),*size,now))continue;
            it->second.last_seen=now;
        }
        for(auto it=peers_.begin();it!=peers_.end();) {
            // Drain completed messages independently of receiving new packets.
            // Backpressure retains acknowledged messages in their bounded
            // channel queue until the application calls pop().
            while(const auto ready=it->second.channel.ready_size()) {
                if(ready>limits_.max_queued-
                   std::min(received_bytes_,limits_.max_queued))break;
                stream::bytes payload;
                if(!it->second.channel.pop(payload))break;
                received_bytes_+=payload.size();
                received_.push_back({it->second.peer,std::move(payload)});
            }
            if(it->second.channel.pending_bytes()==0 &&
               !it->second.channel.ready_size() &&
               now-it->second.last_seen>std::chrono::seconds(60) &&
               it->second.last_seen!=stream::clock::time_point{}) {
                it=peers_.erase(it);
                continue;
            }
            const auto packets=it->second.channel.flush(byte_budget,now);
            for(const auto& packet:packets) {
                if(socket::send(socket_,it->second.peer,packet.data(),packet.size()))
                    byte_budget-=packet.size();
            }
            ++it;
            if(!byte_budget)break;
        }
    }
    bool host::pop(delivery& packet) {
        if(received_.empty())return false;
        received_bytes_-=received_.front().payload.size();
        packet=std::move(received_.front());
        received_.pop_front();
        return true;
    }
}
