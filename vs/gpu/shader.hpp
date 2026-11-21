#pragma once
#include <string>
#include <utility>
#include <vk/vk.hpp>

namespace vs::gpu
{
// Optional shader module owner for callers building custom pipelines.
// Accepts precompiled SPIR-V, not source GLSL/HLSL text.
class shader
{
public:
    shader() = default;
    ~shader() { reset(); }
    shader(const shader &) = delete;
    shader &operator=(const shader &) = delete;
    shader(shader &&other) noexcept : handle_(std::exchange(other.handle_, vk::ShaderModule{})) {}
    shader &operator=(shader &&other) noexcept
    {
        if (this != &other) {
            reset();
            handle_ = std::exchange(other.handle_, vk::ShaderModule{});
        }

        return *this;
    }

    void load(const std::string &spirv_file);
    void reset();
    vk::ShaderModule handle() const { return handle_; }

private:
    vk::ShaderModule handle_{};
};
} // namespace vs::gpu
