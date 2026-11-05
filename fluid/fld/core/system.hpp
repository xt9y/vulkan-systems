#pragma once
#include "types.hpp"
#include <vk/vk.hpp>
#include <array>
#include <cstdint>

namespace fld::system
{
    struct state
    {
        options settings{};
        vk::buffer::resource particles,heads,next,lambdas,deltas,normals,boxes,dummy_bvh;
        vk::DescriptorSetLayout descriptor_layout{};
        vk::DescriptorPool descriptor_pool{};
        vk::DescriptorSet descriptors{};
        vk::PipelineLayout pipeline_layout{};
        std::array<vk::Pipeline,14> pipelines{};
        std::uint32_t count=0;
        std::uint32_t collider_count=0;
        bool bvh_enabled=false,bvh_built=false;
        std::uint32_t bucket_count=16384;
        double simulated_seconds=0.;
    };

    state *get();
    void init(const options &settings,const drop &initial,
              const shader_paths &shaders);
    // Records an entirely GPU-resident fixed-step PBF baseline.
    // The caller owns the graphics queue submission and frame fences.
    void record(vk::CommandBuffer command,float seconds);
    vk::DescriptorBufferInfo descriptor();
    std::uint32_t particle_count();
    void shutdown();
}
