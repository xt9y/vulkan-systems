#pragma once
#include <vk/vk.hpp>
#include <cstdint>
#include <string>
#include <vector>

namespace gpu
{
    // SPIR-V compute stage with caller-owned descriptor layouts and sets.
    // Pipeline and pipeline layout are owned by this object.
    class compute
    {
    public:
        compute()=default;
        ~compute(){reset();}
        compute(const compute&)=delete;
        compute& operator=(const compute&)=delete;

        void create(const std::string& spirv_path,
                    const std::vector<vk::DescriptorSetLayout>& layouts={},
                    std::uint32_t push_bytes=0,
                    const char* entry="main");
        void reset();
        vk::Pipeline handle() const{return pipeline_;}
        vk::PipelineLayout layout() const{return layout_;}
        void dispatch(vk::CommandBuffer command,
                      std::uint32_t x,std::uint32_t y=1,std::uint32_t z=1,
                      const std::vector<vk::DescriptorSet>& sets={},
                      const void* push=nullptr,std::uint32_t push_bytes=0) const;
        void dispatch_indirect(vk::CommandBuffer command,vk::Buffer arguments,
                               vk::DeviceSize offset=0,
                               const std::vector<vk::DescriptorSet>& sets={},
                               const void* push=nullptr,std::uint32_t push_bytes=0) const;
    private:
        void bind(vk::CommandBuffer command,
                  const std::vector<vk::DescriptorSet>& sets,
                  const void* push,std::uint32_t push_bytes) const;
        vk::Pipeline pipeline_{};
        vk::PipelineLayout layout_{};
        std::uint32_t push_limit_=0;
        std::uint32_t descriptor_count_=0;
    };
}
