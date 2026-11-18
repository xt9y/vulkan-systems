#include <vs/lighting/shadow_cache.hpp>
#include <vk/vk.hpp>
#include <array>
#include <stdexcept>
#include <vector>

namespace vs::lighting {
void shadow_cache::create()
{
    if(buffer_.handle)throw std::logic_error("Shadow cache already created");
    buffer_=::vs::gpu::create_buffer(
        std::size_t(slots)*sizeof(std::array<std::uint32_t,4>),
        vk::BufferUsageFlagBits::eStorageBuffer,
        vk::MemoryPropertyFlagBits::eHostVisible |
        vk::MemoryPropertyFlagBits::eHostCoherent);
    invalidate();
}
void shadow_cache::invalidate()
{
    if(!buffer_.handle)return;
    std::vector<std::array<std::uint32_t,4>> cleared(slots);
    vk::buffer::write(buffer_,cleared.data(),
                      cleared.size()*sizeof(cleared.front()));
}
}
