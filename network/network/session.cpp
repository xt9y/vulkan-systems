#include "session.hpp"
#include <algorithm>
#include <array>
#include <stdexcept>
#include <utility>
#include <tuple>

namespace net {
session::session():session(settings{}) {}
session::session(settings options):options_(options) {
    if(!options_.max_peers||!options_.max_delivered_bytes||
       options_.idle_timeout.count()<=0)
        throw std::invalid_argument("invalid session settings");
}
session::clients::iterator session::find_or_open(const endpoint::resource& peer,
                                                 bool unknown) {
    const auto id=endpoint::identifier(peer);
    if(id.empty())return peers_.end();
    auto it=peers_.find(id);
    if(it!=peers_.end())return it;
    if((unknown&&!options_.allow_unknown_peers)||
       peers_.size()>=options_.max_peers)return peers_.end();
    return peers_.emplace(std::piecewise_construct,
        std::forward_as_tuple(id),
        std::forward_as_tuple(peer,options_.channel)).first;
}
bool session::open(const endpoint::resource& peer) {
    return find_or_open(peer,false)!=peers_.end();
}
void session::close(const endpoint::resource& peer) {
    peers_.erase(endpoint::identifier(peer));
}
bool session::send(const endpoint::resource& peer,stream::bytes payload) {
    auto it=find_or_open(peer,false);
    if(it==peers_.end())return false;
    it->second.last_seen=stream::clock::now();
    return it->second.channel.queue(std::move(payload))!=0;
}
bool session::accept(const endpoint::resource& peer,const void* input,
                     std::size_t size,stream::clock::time_point now) {
    if(!input||size<4)return false;
    const auto* p=static_cast<const std::uint8_t*>(input);
    if(p[0]!='V'||p[1]!='S'||p[2]!='N'||p[3]!='1')return false;
    if(size<28||size>options_.channel.mtu)return true;
    auto it=find_or_open(peer,true);
    if(it==peers_.end())return true;
    if(it->second.channel.accept(input,size,now))
        it->second.last_seen=now;
    return true;
}
void session::pump(const socket::resource& udp,std::size_t byte_budget,
                   stream::clock::time_point now) {
    for(auto it=peers_.begin();it!=peers_.end();) {
        auto& client=it->second;
        while(const auto ready=client.channel.ready_size()) {
            if(ready>options_.max_delivered_bytes-
                     std::min(received_bytes_,options_.max_delivered_bytes))break;
            stream::bytes message;
            if(!client.channel.pop(message))break;
            received_bytes_+=message.size();
            received_.push_back({client.peer,std::move(message)});
        }
        // Only reclaim peers when they have no outstanding unacknowledged
        // payloads or already completed messages.
        if(client.channel.pending_bytes()==0 &&
           client.channel.ready_size()==0 &&
           now-client.last_seen>options_.idle_timeout) {
            it=peers_.erase(it);
            continue;
        }
        auto packets=client.channel.flush(byte_budget,now);
        for(const auto& packet:packets) {
            socket::send(udp,client.peer,packet.data(),packet.size());
            byte_budget-=packet.size();
        }
        ++it;
        if(!byte_budget)break;
    }
}
bool session::pop(delivery& out) {
    if(received_.empty())return false;
    received_bytes_-=received_.front().payload.size();
    out=std::move(received_.front());
    received_.pop_front();
    return true;
}
std::size_t session::pending(const endpoint::resource& peer) const {
    const auto it=peers_.find(endpoint::identifier(peer));
    return it==peers_.end()?0:it->second.channel.pending_bytes();
}
void session::clear() noexcept {
    peers_.clear();
    received_.clear();
    received_bytes_=0;
}
}
