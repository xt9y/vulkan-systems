#pragma once
#include <vk/vk.hpp>
#include <string>

namespace gpu
{
    // Optional shader module owner for callers building custom pipelines.
    // Accepts precompiled SPIR-V, not source GLSL/HLSL text.
    class shader
    {
    public:
        shader()=default;
        ~shader(){reset();}
        shader(const shader&)=delete;
        shader& operator=(const shader&)=delete;
        void load(const std::string& spirv_file);
        void reset();
        vk::ShaderModule handle() const{return handle_;}
    private:
        vk::ShaderModule handle_{};
    };
}
