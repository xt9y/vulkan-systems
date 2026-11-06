#include "endpoint.hpp"
#include <stdexcept>
#include <cstring>
#ifndef _WIN32
#include <netdb.h>
#endif
namespace net::endpoint {
resource resolve(const std::string& host,std::uint16_t port) {
    addrinfo hints{};
    hints.ai_family=AF_UNSPEC;
    hints.ai_socktype=SOCK_DGRAM;
    hints.ai_protocol=IPPROTO_UDP;
    addrinfo* addresses=nullptr;
    const auto service=std::to_string(port);
    const int error=getaddrinfo(host.c_str(),service.c_str(),&hints,&addresses);
    if(error!=0||!addresses)
        throw std::runtime_error("Cannot resolve UDP endpoint: "+host);
    resource result{};
    for(auto* p=addresses;p;p=p->ai_next) {
        if((p->ai_family!=AF_INET&&p->ai_family!=AF_INET6) ||
           p->ai_addrlen>sizeof(result.address))continue;
        std::memcpy(&result.address,p->ai_addr,p->ai_addrlen);
        result.length=static_cast<decltype(result.length)>(p->ai_addrlen);
        break;
    }
    freeaddrinfo(addresses);
    if(!result.length)throw std::runtime_error("Unsupported UDP address: "+host);
    return result;
}
std::string identifier(const resource& value) {
    const auto family=value.address.ss_family;
    std::string id;
    if(family==AF_INET) {
        const auto& a=reinterpret_cast<const sockaddr_in&>(value.address);
        id.push_back('4');
        id.append(reinterpret_cast<const char*>(&a.sin_port),sizeof(a.sin_port));
        id.append(reinterpret_cast<const char*>(&a.sin_addr),sizeof(a.sin_addr));
    }else if(family==AF_INET6) {
        const auto& a=reinterpret_cast<const sockaddr_in6&>(value.address);
        id.push_back('6');
        id.append(reinterpret_cast<const char*>(&a.sin6_port),sizeof(a.sin6_port));
        id.append(reinterpret_cast<const char*>(&a.sin6_addr),sizeof(a.sin6_addr));
        id.append(reinterpret_cast<const char*>(&a.sin6_scope_id),sizeof(a.sin6_scope_id));
    }
    return id;
}
bool equal(const resource& a,const resource& b) {
    const auto lhs=identifier(a);
    return !lhs.empty()&&lhs==identifier(b);
}
}
