#pragma once
#include <vs/gpu/buffer.hpp>
#include <vs/gpu/compute.hpp>
#include <vs/pmap/pmap.hpp>
#include <vs/gpu/descriptors.hpp>
#include <vk/vk.hpp>
#include <cstdint>
#include <memory>
#include <vector>

namespace vs::optics {
using photons=::vs::pmap::photon_output;
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
                const char* shader="build/shaders/optics_project.comp.spv");
    void set_photons(photons source);
    void record(vk::CommandBuffer command,std::uint32_t image,
                const vk::util::math::mat4& view_projection);
    vk::Buffer buffer(std::uint32_t image) const;
    void reset() noexcept;
private:
    vk::Extent2D extent_{};
    std::vector<std::unique_ptr<::vs::gpu::buffer>> outputs_;
    ::vs::gpu::compute program_;
    ::vs::gpu::descriptors sets_;
    vk::Sampler sampler_{};
    photons source_{};
};
}
