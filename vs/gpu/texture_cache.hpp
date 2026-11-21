#pragma once
#include <cstdint>
#include <unordered_map>
#include <vector>
#include <vs/gpu/image.hpp>
#include <vs/gpu/upload.hpp>

namespace vs::gpu
{
// RGBA8 GPU texture residency/cache; keys and source hashes are application-
// supplied. Callers own the sampler and must synchronize replacements.
class texture_cache
{
public:
    texture_cache() = default;
    ~texture_cache() { clear(); }
    texture_cache(const texture_cache &) = delete;
    texture_cache &operator=(const texture_cache &) = delete;
    bool contains(std::uint64_t key, std::uint32_t hash) const noexcept;
    void upsert(std::uint64_t key, std::uint32_t hash,
                const void *rgba, std::uint32_t width, std::uint32_t height,
                upload::batch &transfers);
    void adopt(std::uint64_t key, std::uint32_t hash,
               std::uint32_t width, std::uint32_t height, std::uint32_t levels,
               vk::image::resource resource);
    vk::ImageView view(std::uint64_t key) const;
    void retain(const std::vector<std::uint64_t> &keys);
    void clear() noexcept;
    std::size_t size() const noexcept { return images_.size(); }

private:
    struct texture
    {
        std::uint32_t hash = 0, width = 0, height = 0;
        gpu::image image;
    };

    std::unordered_map<std::uint64_t, texture> images_;
};
} // namespace vs::gpu
