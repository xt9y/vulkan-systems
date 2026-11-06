#include "host.hpp"
#include <array>
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
        active_=false; peers_.clear(); received_.clear();
    }
    bool host::send(const endpoint::resource& peer,stream::bytes payload) {
        if(!active_)return false;
        const auto id=key(peer);
        auto it=peers_.find(id);
        if(it==peers_.end()) {
            if(peers_.size()>=256)return false;
            it=peers_.emplace(
                std::piecewise_construct,
                std::forward_as_tuple(id),
                std::forward_as_tuple(peer,limits_)).first;
        }
        return it->second.channel.queue(std::move(payload))!=0;
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
            stream::bytes payload;
            while(it->second.channel.pop(payload))
                received_.push_back({sender,std::move(payload)});
        }
        for(auto it=peers_.begin();it!=peers_.end();) {
            if(it->second.channel.pending_bytes()==0 &&
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
        packet=std::move(received_.front());
        received_.pop_front();
        return true;
    }
}
