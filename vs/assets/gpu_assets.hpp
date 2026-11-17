#pragma once
#include <vs/scene/scene.hpp>
#include <vs/gpu/texture_cache.hpp>
#include <vs/render/geometry.hpp>
#include <cstdint>
#include <unordered_set>
#include <span>
#include <vector>
namespace vs {
// GPU-side glTF asset residency. Does not dictate a material shader or layout.
class gpu_assets {
public:
    struct mesh {
        std::uint32_t id=0,skin_id=0;
        ::vs::scene::material material{};
        ::vs::render::geometry geometry;
    };
    gpu_assets()=default;
    ~gpu_assets(){reset();}
    gpu_assets(const gpu_assets&)=delete;
    gpu_assets& operator=(const gpu_assets&)=delete;
    // Requires initialized Vulkan device. Caller ensures GPU idle before
    // replacing meshes/textures referenced by in-flight command buffers.
    void upload(std::span<const scene::mesh> meshes,
                std::span<const scene::texture> textures);
    const std::vector<mesh>& meshes() const noexcept {return meshes_;}
    const mesh* find(std::uint32_t id) const noexcept;
    vk::ImageView texture(std::uint32_t id) const;
    vk::Sampler sampler() const noexcept {return sampler_;}
    // Expects five combined image sampler bindings (default 4..8).
    void bind(vk::DescriptorSet set,const ::vs::scene::material& material,
              std::uint32_t first_binding=4) const;
    void reset() noexcept; // call before Vulkan device shutdown
private:
    void initialize();
    std::vector<mesh> meshes_;
    ::vs::gpu::texture_cache cache_;
    std::unordered_set<std::uint32_t> live_;
    vk::Sampler sampler_{};
};
}
