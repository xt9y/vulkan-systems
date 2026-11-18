#pragma once
#include <vs/gpu/resource.hpp>
#include <cstddef>
#include <cstdint>

namespace vs::lighting {
// World-space cache of static opaque shadow visibility. The GPU writes only
// fixed-size 32-bit atomic entries; moving actor shadows remain uncached.
class shadow_cache {
public:
    static constexpr std::uint32_t slots=1u<<17u;
    void create();
    // Caller must wait for GPU submissions that use the buffer before clearing.
    void invalidate();
    void reset() noexcept {buffer_.reset();}
    vk::Buffer buffer() const noexcept {return buffer_.handle;}
    const ::vs::gpu::owned_buffer& resource() const noexcept {return buffer_;}
private:
    ::vs::gpu::owned_buffer buffer_;
};
}
