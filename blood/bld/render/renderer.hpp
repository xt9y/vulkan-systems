#pragma once
#include "../core/types.hpp"
#include <fld/render/renderer.hpp>

namespace bld::render
{
    using input=fld::render::input;

    struct state
    {
        bool owns_renderer=false;
    };

    state *get();
    void init(const std::string &shader_spirv,
              std::uint32_t frames,std::uint32_t width,
              std::uint32_t height);
    void record(vk::CommandBuffer command,std::uint32_t frame,
                const input &inputs);
    bool active();
    void shutdown();
}
