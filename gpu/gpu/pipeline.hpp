#pragma once
#include <vk/vk.hpp>
#include <vector>

namespace gpu
{
    // Generic layout for any graphics/compute pipeline, independent of shaders.
    class pipeline_layout
    {
    public:
        pipeline_layout()=default;
        ~pipeline_layout(){reset();}
        pipeline_layout(const pipeline_layout&)=delete;
        pipeline_layout& operator=(const pipeline_layout&)=delete;
        void create(const std::vector<vk::DescriptorSetLayout>& sets={},
                    const std::vector<vk::PushConstantRange>& ranges={});
        void reset();
        vk::PipelineLayout handle() const{return handle_;}
    private:
        vk::PipelineLayout handle_{};
    };
}
