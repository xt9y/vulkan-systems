#pragma once
#include <cstdint>
#include <string>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <sys/socket.h>
#include <netinet/in.h>
#endif
namespace net::endpoint {
struct resource {
    sockaddr_storage address{};
#ifdef _WIN32
    int length=0;
#else
    socklen_t length=0;
#endif
};
resource resolve(const std::string& host,std::uint16_t port);
bool equal(const resource& first,const resource& second);
// Collision-free family/address/port identity, including IPv6 scope ID.
std::string identifier(const resource& peer);
}
