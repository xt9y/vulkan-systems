#include "lumen.hpp"
#include <utility>

namespace lumen {

struct system::implementation {
    settings options{};
    vk::RenderPass presentation_pass{};
    vk::Format presentation_format=vk::Format::eUndefined;
    vk::Extent2D extent{};
    scene geometry{};
    lighting lights{};
    gbuffer surfaces{};
};

// These are deliberately inert. No placeholder rendering or fake GI values.
system::system()=default;
system::~system()=default;
system::system(system&&) noexcept=default;
system& system::operator=(system&&) noexcept=default;

void system::create(vk::RenderPass pass, vk::Format format,
                    vk::Extent2D extent, const settings& options)
{
    impl_=std::make_unique<implementation>();
    impl_->options=options;
    impl_->presentation_pass=pass;
    impl_->presentation_format=format;
    impl_->extent=extent;
}

void system::resize(vk::RenderPass pass, vk::Format format, vk::Extent2D extent)
{
    if(!impl_)return;
    impl_->presentation_pass=pass;
    impl_->presentation_format=format;
    impl_->extent=extent;
}

void system::set_scene(const scene& value)
{
    if(impl_)impl_->geometry=value;
}

void system::set_lighting(const lighting& value)
{
    if(impl_)impl_->lights=value;
}

void system::set_gbuffer(const gbuffer& value)
{
    if(impl_)impl_->surfaces=value;
}

void system::invalidate()
{
    // TODO(lumen): mark screen probes, surface cards and radiance cache dirty.
}

void system::record(vk::CommandBuffer, const frame&)
{
    // TODO(lumen): screen trace, scene trace, surface/radiance cache,
    // probe gather, indirect illumination and reflection passes.
}

void system::compose(vk::CommandBuffer)
{
    // TODO(lumen): composite the lighting result into an active host pass.
}

outputs system::result() const noexcept
{
    return {};
}

bool system::implemented() const noexcept
{
    return false;
}

void system::reset() noexcept
{
    impl_.reset();
}

} // namespace lumen
