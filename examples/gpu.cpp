#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <vector>
#include <vs/app/app.hpp>
#include <vs/gpu/gpu.hpp>

// GPU-only particle simulation: one user-owned buffer, one descriptor set,
// one compute program, and an explicit synchronization graph.
int main()
{
    constexpr std::uint32_t count = 4096;
    vs::application app({.title = "vs / gpu - particles", .frames_in_flight = 1});
    std::vector<std::array<float, 4>> positions(count);

    for (std::uint32_t i = 0; i < count; ++i)
    {
        const float x = float(i % 64) / 32.f - 1.f, y = float(i / 64) / 32.f - 1.f;
        positions[i] = {x, y, 0, 1};
    }

    vs::gpu::buffer particles;
    particles.create(sizeof(positions[0]) * positions.size(),
                     vk::BufferUsageFlagBits::eStorageBuffer,
                     vk::MemoryPropertyFlagBits::eHostVisible |
                         vk::MemoryPropertyFlagBits::eHostCoherent);
    particles.upload(positions.data(), sizeof(positions[0]) * positions.size());

    vk::DescriptorSetLayoutBinding binding{};
    binding.binding = 0;
    binding.descriptorCount = 1;
    binding.descriptorType = vk::DescriptorType::eStorageBuffer;
    binding.stageFlags = vk::ShaderStageFlagBits::eCompute;
    vs::gpu::descriptors sets;
    sets.create({binding});
    sets.write_buffer(0, 0, particles.descriptor());
    vs::gpu::compute simulation;
    simulation.create("build/shaders/example_particles.comp.spv",
                      {sets.layout()}, sizeof(float));

    vs::gpu::graph graph;

    const auto start = std::chrono::steady_clock::now();

    app.run([&](vk::CommandBuffer cmd, std::uint32_t image)
            {
        const float t=std::chrono::duration<float>(
            std::chrono::steady_clock::now()-start).count();
        graph.clear_passes();
        graph.add("particle integration",{
            vs::gpu::access::for_buffer(particles.handle(),
                vk::PipelineStageFlagBits::eComputeShader,
                vk::AccessFlagBits::eShaderWrite,true)
        },[&](vk::CommandBuffer command) {
            simulation.dispatch(command,(count+63)/64,1,1,{sets.set()},&t,sizeof(t));
        });
        graph.execute(cmd);
        graph.clear_passes();
        app.clear(cmd,image,{.04f+.025f*std::sin(t),.055f,.10f,1.f}); });
    vk::device::idle();
    simulation.reset();
    sets.reset();
    particles.reset();
    app.close();
}
