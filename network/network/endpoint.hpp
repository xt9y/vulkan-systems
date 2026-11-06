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
#include <netinet/in.h>
#endif

namespace net::endpoint
{
    struct resource
    {
        sockaddr_in address{};
    };

    resource resolve(const std::string& host, std::uint16_t port);
    bool equal(const resource& first, const resource& second);
} // namespace net::endpoint
