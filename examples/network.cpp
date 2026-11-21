#include <array>
#include <chrono>
#include <cstring>
#include <iostream>
#include <vs/network/endpoint.hpp>
#include <vs/network/socket.hpp>

// In-process UDP echo; each socket is owned by this example. No engine state,
// DTLS keys, or central networking singleton scene is required.
int main()
{
    using namespace vs::network;
    socket::init();

    auto server = socket::create(47602, "127.0.0.1");
    auto client = socket::create(0, "127.0.0.1");
    const auto destination = endpoint::resolve("127.0.0.1", 47602);

    constexpr char message[] = "hello from vs";

    const bool sent = socket::send(client, destination, message, sizeof(message) - 1);

    socket::wait_readable(server, std::chrono::milliseconds(1000));
    std::array<char, 128> buffer{};
    endpoint::resource remote{};

    const auto received = socket::receive(server, remote, buffer.data(), buffer.size());

    if (sent && received) {
        socket::send(server, remote, buffer.data(), *received);
        socket::wait_readable(client, std::chrono::milliseconds(1000));
        endpoint::resource responder{};

        const auto echoed = socket::receive(client, responder, buffer.data(), buffer.size());

        if (echoed) std::cout << "UDP loopback: " << std::string(buffer.data(), *echoed) << '\n';
    }

    socket::destroy(client);
    socket::destroy(server);
    socket::shutdown();
}
