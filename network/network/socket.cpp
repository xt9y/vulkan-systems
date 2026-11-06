#include "socket.hpp"
#include <stdexcept>
#include <limits>
#ifdef _WIN32
namespace {
bool retryable_error() {
    const auto e=WSAGetLastError();
    return e==WSAEWOULDBLOCK||e==WSAEINTR||e==WSAECONNRESET||
        e==WSAENETUNREACH||e==WSAEHOSTUNREACH||e==WSAENOBUFS||e==WSAENETDOWN;
}
}
#else
#include <cerrno>
#include <fcntl.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>
namespace {
bool retryable_error() {
    return errno==EAGAIN||errno==EWOULDBLOCK||errno==EINTR||
        errno==ECONNREFUSED||errno==ENETUNREACH||
        errno==EHOSTUNREACH||errno==ENOBUFS||errno==ENETDOWN;
}
}
#endif
namespace net::socket {
state* get(){static state data;return &data;}
void init(){
    if(get()->initialized)return;
#ifdef _WIN32
    WSADATA data{};
    if(WSAStartup(MAKEWORD(2,2),&data))
        throw std::runtime_error("WSAStartup failed");
#endif
    get()->initialized=true;
}
void shutdown() noexcept {
    if(!get()->initialized)return;
#ifdef _WIN32
    WSACleanup();
#endif
    *get()={};
}
resource create(std::uint16_t port,const std::string& bind_host) {
    if(!get()->initialized)
        throw std::runtime_error("net::socket runtime is not initialized");
    const auto local=endpoint::resolve(bind_host,port);
    resource result{};
    result.handle=::socket(local.address.ss_family,SOCK_DGRAM,IPPROTO_UDP);
#ifdef _WIN32
    if(result.handle==INVALID_SOCKET)throw std::runtime_error("Cannot create UDP socket");
    u_long enabled=1;
    const bool configured=ioctlsocket(result.handle,FIONBIO,&enabled)==0;
#else
    if(result.handle<0)throw std::runtime_error("Cannot create UDP socket");
    const auto flags=fcntl(result.handle,F_GETFL,0);
    const bool configured=flags>=0&&fcntl(result.handle,F_SETFL,flags|O_NONBLOCK)==0;
#endif
    // Dual stack where the OS permits it. Explicit IPv4/IPv6 binds still work.
    if(local.address.ss_family==AF_INET6) {
        const int off=0;
        ::setsockopt(result.handle,IPPROTO_IPV6,IPV6_V6ONLY,
                     reinterpret_cast<const char*>(&off),sizeof(off));
    }
    const int receive_bytes=2*1024*1024;
    ::setsockopt(result.handle,SOL_SOCKET,SO_RCVBUF,
        reinterpret_cast<const char*>(&receive_bytes),sizeof(receive_bytes));
    if(!configured||::bind(result.handle,
         reinterpret_cast<const sockaddr*>(&local.address),local.length)!=0) {
#ifdef _WIN32
        closesocket(result.handle);
#else
        ::close(result.handle);
#endif
        throw std::runtime_error("Cannot configure/bind UDP socket");
    }
    return result;
}
void destroy(resource& target) noexcept {
#ifdef _WIN32
    if(target.handle!=INVALID_SOCKET)closesocket(target.handle);
#else
    if(target.handle>=0)::close(target.handle);
#endif
    target={};
}
void wait_readable(const resource& target,std::chrono::milliseconds timeout) {
#ifdef _WIN32
    WSAPOLLFD descriptor{target.handle,POLLRDNORM,0};
    const auto result=WSAPoll(&descriptor,1,static_cast<int>(timeout.count()));
#else
    pollfd descriptor{target.handle,POLLIN,0};
    const auto result=::poll(&descriptor,1,static_cast<int>(timeout.count()));
#endif
    if(result<0&&!retryable_error())throw std::runtime_error("UDP poll failed");
}
bool send(const resource& target,const endpoint::resource& peer,const void* data,std::size_t size) {
    if(!data||!peer.length||size>static_cast<std::size_t>(std::numeric_limits<int>::max()))
        return false;
    const auto sent=::sendto(target.handle,static_cast<const char*>(data),
        static_cast<int>(size),0,
        reinterpret_cast<const sockaddr*>(&peer.address),peer.length);
    if(sent<0) {
        if(retryable_error())return false;
        throw std::runtime_error("UDP send failed");
    }
    return static_cast<std::size_t>(sent)==size;
}
std::optional<std::size_t> receive(const resource& target,endpoint::resource& peer,
                                   void* data,std::size_t capacity) {
    if(!data||capacity>static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("invalid UDP receive capacity");
#ifdef _WIN32
    int length=sizeof(sockaddr_storage);
#else
    socklen_t length=sizeof(sockaddr_storage);
#endif
    const auto received=::recvfrom(target.handle,static_cast<char*>(data),
        static_cast<int>(capacity),0,reinterpret_cast<sockaddr*>(&peer.address),&length);
    if(received<0) {
#ifdef _WIN32
        if(WSAGetLastError()==WSAEMSGSIZE)return capacity;
#endif
        if(retryable_error())return std::nullopt;
        throw std::runtime_error("UDP receive failed");
    }
    peer.length=length;
    return static_cast<std::size_t>(received);
}
}
