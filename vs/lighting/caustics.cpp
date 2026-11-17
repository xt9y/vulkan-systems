#include <vs/lighting/caustics.hpp>
#include <algorithm>
#include <stdexcept>

namespace vs::lighting {
void refractive_caustics::create(vk::Extent2D extent,
    const std::vector<vk::ImageView>& depth,std::uint32_t photons)
{
    if(!extent.width || !extent.height || depth.empty() ||
       photons<64u || photons>65536u)
        throw std::invalid_argument("lighting: invalid caustic target");
    reset();
    extent_=extent;
    photons_=photons;
    try {projector_.create(extent,depth);}
    catch(...) {reset();throw;}
}
void refractive_caustics::set_scene(const ::vs::scene::data& source)
{
    if(!extent_.width)throw std::logic_error("lighting: caustics not created");
    transport_.reset();
    bool glass=false,light=false;
    for(const auto& f:source.surfaces)
        glass|=f.optical_type==1u && f.optical_strength>0.f;
    ::vs::scene::each_mesh_surface(source,[&](const auto& f) {
        glass|=f.optical_type==1u && f.optical_strength>0.f;
    });
    for(const auto& lamp:source.lights)
        light|=lamp.enabled && lamp.strength>0.f && lamp.radius>0.f;
    if(!glass || !light) {
        projector_.set_photons({});
        return;
    }
    ::vs::pmap::configuration cfg{};
    cfg.photons=photons_;
    cfg.bounces=4;
    cfg.gi_updates=1;
    cfg.transport_only=true;
    auto next=std::make_unique<::vs::pmap::mapper>();
    next->create(source,extent_,cfg);
    projector_.set_photons(next->photon_buffers());
    transport_=std::move(next);
}
void refractive_caustics::record(vk::CommandBuffer cmd,
    std::uint32_t image,const vk::util::math::mat4& view_projection)
{
    if(transport_)transport_->record_transport(cmd);
    projector_.record(cmd,image,view_projection);
}
vk::Buffer refractive_caustics::buffer(std::uint32_t image) const
{
    return projector_.buffer(image);
}
void refractive_caustics::reset() noexcept
{
    transport_.reset();
    projector_.reset();
    extent_=vk::Extent2D{};
}
}
