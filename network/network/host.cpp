#include "host.hpp"
#include <array>
#include <stdexcept>
#include <utility>

namespace net {
host::host(stream::settings limits)
    :channels_([&]{session::settings o;o.channel=limits;return o;}()) {}
void host::start(std::uint16_t port,const std::string& bind) {
    if(active_)throw std::logic_error("network host already started");
    socket::init();
    socket_=socket::create(port,bind);
    active_=true;
}
void host::stop() noexcept {
    if(active_)socket::destroy(socket_);
    active_=false;
    channels_.clear();
}
bool host::send(const endpoint::resource& peer,stream::bytes payload) {
    if(!active_)return false;
    return channels_.send(peer,std::move(payload));
}
void host::pump(std::size_t receives,std::size_t byte_budget) {
    if(!active_)throw std::logic_error("network host is not started");
    std::array<std::uint8_t,1473> packet{};
    const auto now=stream::clock::now();
    for(std::size_t i=0;i<receives;++i) {
        endpoint::resource sender;
        const auto size=socket::receive(socket_,sender,packet.data(),packet.size());
        if(!size)break;
        channels_.accept(sender,packet.data(),*size,now);
    }
    channels_.pump(socket_,byte_budget,now);
}
bool host::pop(delivery& value){return channels_.pop(value);}
}
