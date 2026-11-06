#pragma once
#include "session.hpp"
#include <cstddef>
#include <cstdint>
#include <string>

namespace net {
// Dedicated-socket adapter for the same reusable transport used by session.
class host {
public:
    explicit host(stream::settings limits={});
    ~host(){stop();}
    host(const host&)=delete;
    host& operator=(const host&)=delete;
    void start(std::uint16_t port=0,const std::string& bind="0.0.0.0");
    void stop() noexcept;
    bool send(const endpoint::resource& peer,stream::bytes payload);
    void pump(std::size_t max_receives=256,std::size_t byte_budget=256*1024);
    bool pop(delivery& received);
private:
    session channels_;
    socket::resource socket_{};
    bool active_=false;
};
}
