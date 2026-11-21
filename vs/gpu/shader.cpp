#include <stdexcept>
#include <vs/gpu/shader.hpp>

namespace vs::gpu
{
void shader::load(const std::string &path)
{
    if (handle_) throw std::logic_error("shader already initialized");
    if (path.empty()) throw std::invalid_argument("empty SPIR-V path");
    handle_ = vk::shader::create(vk::shader::read(path));
}

void shader::reset()
{
    if (handle_) vk::shader::destroy(handle_);
    handle_ = vk::ShaderModule{};
}
} // namespace vs::gpu
