#include "socket.hpp"

#include <stdexcept>
#ifdef _WIN32
namespace
{
    bool retryable_error()
    {
        const auto e = WSAGetLastError();

        return e == WSAEWOULDBLOCK || e == WSAEINTR || e == WSAECONNRESET || e == WSAENETUNREACH ||
               e == WSAEHOSTUNREACH || e == WSAENOBUFS || e == WSAENETDOWN;
    }
} // namespace
#else
#include <cerrno>
#include <fcntl.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

namespace
{
    bool retryable_error()
    {
        return errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR || errno == ECONNREFUSED ||
               errno == ENETUNREACH || errno == EHOSTUNREACH || errno == ENOBUFS ||
               errno == ENETDOWN;
    }
} // namespace
#endif

namespace net::socket
{
    state* get()
    {
        static state data;
        return &data;
    }

    void init()
    {
        if (get()->initialized)
        {
            return;
        }

#ifdef _WIN32
        WSADATA data{};
        if (WSAStartup(MAKEWORD(2, 2), &data))
        {
            throw std::runtime_error("WSAStartup failed");
        }
#endif
        get()->initialized = true;
    }

    void shutdown() noexcept
    {
        if (!get()->initialized)
        {
            return;
        }

#ifdef _WIN32
        WSACleanup();
#endif
        *get() = {};
    }

    resource create(std::uint16_t port, const std::string& bind_host)
    {
        if (!get()->initialized)
        {
            throw std::runtime_error("net::socket runtime is not initialized");
        }

        resource result;

        const auto local = endpoint::resolve(bind_host, port);
        result.handle = ::socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
#ifdef _WIN32
        if (result.handle == INVALID_SOCKET)
        {
            throw std::runtime_error("Cannot create UDP socket");
        }

        u_long enabled = 1;
        const bool configured = ioctlsocket(result.handle, FIONBIO, &enabled) == 0;
#else
        if (result.handle < 0)
        {
            throw std::runtime_error("Cannot create UDP socket");
        }

        const auto flags = fcntl(result.handle, F_GETFL, 0);
        const bool configured =
            flags >= 0 && fcntl(result.handle, F_SETFL, flags | O_NONBLOCK) == 0;
#endif

        // Scene streaming can arrive in bursts. Keep enough kernel receive
        // capacity to avoid losing a full window before the network worker drains it.
        const int receive_bytes = 2 * 1024 * 1024;
        ::setsockopt(result.handle, SOL_SOCKET, SO_RCVBUF,
                     reinterpret_cast<const char*>(&receive_bytes),
                     sizeof(receive_bytes));

        if (!configured || ::bind(
                               result.handle,
                               reinterpret_cast<const sockaddr*>(&local.address),
                               sizeof(sockaddr_in)
                           ) != 0)
        {
#ifdef _WIN32
            closesocket(result.handle);
#else
            ::close(result.handle);
#endif

            throw std::runtime_error(
                "Cannot configure/bind UDP socket (port may already be in use)"
            );
        }
        return result;
    }

    void destroy(resource& socket) noexcept
    {
#ifdef _WIN32
        if (socket.handle != INVALID_SOCKET)
        {
            closesocket(socket.handle);
        }
#else
        if (socket.handle >= 0)
        {
            ::close(socket.handle);
        }
#endif
        socket = {};
    }

    void wait_readable(const resource& socket, std::chrono::milliseconds timeout)
    {
#ifdef _WIN32
        WSAPOLLFD descriptor{socket.handle, POLLRDNORM, 0};
        const auto result = WSAPoll(&descriptor, 1, static_cast<int>(timeout.count()));
#else
        pollfd descriptor{socket.handle, POLLIN, 0};
        const auto result = ::poll(&descriptor, 1, static_cast<int>(timeout.count()));
#endif

        if (result < 0 && !retryable_error())
        {
            throw std::runtime_error("UDP poll failed");
        }
    }

    bool
    send(const resource& socket, const endpoint::resource& peer, const void* data, std::size_t size)
    {
        const auto sent = ::sendto(
            socket.handle,
            static_cast<const char*>(data),
            static_cast<int>(size),
            0,
            reinterpret_cast<const sockaddr*>(&peer.address),
            sizeof(sockaddr_in)
        );

        if (sent < 0)
        {
            if (retryable_error())
            {
                return false;
            }

            throw std::runtime_error("UDP send failed");
        }

        return static_cast<std::size_t>(sent) == size;
    }

    std::optional<std::size_t>
    receive(const resource& socket, endpoint::resource& peer, void* data, std::size_t capacity)
    {
#ifdef _WIN32
        int length = sizeof(sockaddr_in);
#else
        socklen_t length = sizeof(sockaddr_in);
#endif

        const auto received = ::recvfrom(
            socket.handle,
            static_cast<char*>(data),
            static_cast<int>(capacity),
            0,
            reinterpret_cast<sockaddr*>(&peer.address),
            &length
        );

        if (received < 0)
        {
#ifdef _WIN32
            if (WSAGetLastError() == WSAEMSGSIZE)
            {
                return capacity; // Discard truncated packets in the decoder.
            }
#endif

            if (retryable_error())
            {
                return std::nullopt;
            }

            throw std::runtime_error("UDP receive failed");
        }

        return static_cast<std::size_t>(received);
    }
} // namespace net::socket
