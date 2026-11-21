#pragma once
#include <cstdint>
#include <memory>
#include <span>
#include <unordered_set>
#include <vector>
#include <vs/gpu/texture_cache.hpp>
#include <vs/render/geometry.hpp>
#include <vs/scene/scene.hpp>
namespace vs
{
// GPU-side glTF asset residency. Does not dictate a material shader or layout.
class gpu_assets
{
public:
    struct mesh
    {
        std::uint32_t id = 0, skin_id = 0;
        ::vs::scene::material material{};
        ::vs::render::geometry geometry;
    };
    // Enable anisotropy only if the caller enabled samplerAnisotropy while
    // creating its logical device. Otherwise keep ordinary trilinear sampling.
    explicit gpu_assets(bool anisotropy_enabled = false);
    ~gpu_assets();
    gpu_assets(const gpu_assets &) = delete;
    gpu_assets &operator=(const gpu_assets &) = delete;
    // Requires initialized Vulkan device. Caller ensures GPU idle before
    // replacing meshes/textures referenced by in-flight command buffers.
    void upload(std::span<const scene::mesh> meshes,
                std::span<const scene::texture> textures);
    // Add immutable streamed textures without rescanning the cumulative catalog.
    // Incrementally stage decoded texture rows, returning true when all
    // textures in this batch are resident and their transfer has completed.
    bool append_textures(std::span<const scene::texture> textures);
    bool append_textures(std::span<const scene::texture> textures,
                         std::span<const std::uint32_t> hashes);
    void cancel_texture_append();
    const std::vector<mesh> &meshes() const noexcept { return meshes_; }
    const mesh *find(std::uint32_t id) const noexcept;
    vk::ImageView texture(std::uint32_t id) const;
    vk::Sampler sampler() const noexcept { return sampler_; }
    // Expects five combined image sampler bindings (default 4..8).
    void bind(vk::DescriptorSet set, const ::vs::scene::material &material,
              std::uint32_t first_binding = 4) const;
    void reset() noexcept; // call before Vulkan device shutdown
private:
    struct texture_append_state;

    void initialize();
    std::vector<mesh> meshes_;
    ::vs::gpu::texture_cache cache_;
    std::unordered_set<std::uint32_t> live_;
    vk::Sampler sampler_{};

    bool anisotropy_enabled_ = false;

    std::unique_ptr<texture_append_state> texture_append_;
};
} // namespace vs
