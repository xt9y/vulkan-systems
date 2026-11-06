#include "endpoint.hpp"

#include <stdexcept>
#ifndef _WIN32
#include <netdb.h>
#endif

namespace net::endpoint
{
    resource resolve(const std::string& host, std::uint16_t port)
    {
        addrinfo hints{};
        hints.ai_family = AF_INET;
        hints.ai_socktype = SOCK_DGRAM;

        addrinfo* addresses = nullptr;
        const int error = getaddrinfo(host.c_str(), nullptr, &hints, &addresses);

        if (error != 0)
        {
            throw std::runtime_error("Cannot resolve UDP endpoint: " + host);
        }

        resource result;
        result.address = *reinterpret_cast<sockaddr_in*>(addresses->ai_addr);
        result.address.sin_port = htons(port);
        freeaddrinfo(addresses);

        return result;
    }

    bool equal(const resource& first, const resource& second)
    {
        return first.address.sin_addr.s_addr == second.address.sin_addr.s_addr &&
               first.address.sin_port == second.address.sin_port;
    }
} // namespace net::endpoint
