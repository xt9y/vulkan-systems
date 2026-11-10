#pragma once
#include <gpu/buffer.hpp>
#include <gpu/compute.hpp>
#include <gpu/descriptors.hpp>
#include <vk/vk.hpp>
#include <cstdint>
#include <memory>
#include <vector>

namespace optics {
struct photons {
    vk::Buffer sorted{},used{};
    std::uint32_t capacity=0;
};
// Projects only marked specular->diffuse photon deposits into the camera's
// visible opaque depth. Atomic fixed-point radiance avoids float-atomic
// extensions on MoltenVK and needs no hardware ray tracing.
class caustics {
public:
    caustics()=default;
    ~caustics(){reset();}
    caustics(const caustics&)=delete;
    caustics& operator=(const caustics&)=delete;
    void create(vk::Extent2D extent,const std::vector<vk::ImageView>& depth,
                const std::vector<vk::ImageView>& hdr,
                const char* shader="build/shaders/optics_project.comp.spv",
                const char* resolve="build/shaders/optics_caustic_resolve.comp.spv");
    void set_photons(photons source);
    void record(vk::CommandBuffer command,std::uint32_t image,
                const vk::util::math::mat4& view_projection);
    vk::Buffer buffer(std::uint32_t image) const;
    void reset() noexcept;
private:
    vk::Extent2D extent_{};
    std::vector<std::unique_ptr<gpu::buffer>> outputs_;
    gpu::compute program_, resolve_;
    gpu::descriptors sets_, resolve_sets_;
    vk::Sampler sampler_{};
    photons source_{};
};
}
