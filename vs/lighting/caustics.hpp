#pragma once
#include <vs/optics/caustics.hpp>
#include <vs/pmap/pmap.hpp>
#include <vs/scene/scene.hpp>
#include <vk/vk.hpp>
#include <cstdint>
#include <memory>
#include <vector>

namespace vs::lighting {
// Optional physically refracted photon-caustic pass. The host decides when
// to enable it; ordinary PCloud and glass shading require none of this work.
// Reuses PMap's Snell/Fresnel photon transport and the existing VS depth-
// checked caustic projector, not another gameplay-specific renderer.
class refractive_caustics {
public:
    void create(vk::Extent2D extent,const std::vector<vk::ImageView>& depth,
                std::uint32_t photons=4096);
    // Device submissions must be idle when replacing the scene or targets.
    void set_scene(const ::vs::scene::data&);
    void record(vk::CommandBuffer,std::uint32_t image,
                const vk::util::math::mat4& view_projection);
    vk::Buffer buffer(std::uint32_t image) const;
    bool active() const noexcept {return static_cast<bool>(transport_);}
    void reset() noexcept;
private:
    vk::Extent2D extent_{};
    ::vs::optics::caustics projector_;
    std::unique_ptr<::vs::pmap::mapper> transport_;
    std::uint32_t photons_=4096;
};
}
