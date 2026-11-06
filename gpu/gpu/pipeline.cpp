#include "pipeline.hpp"
#include <stdexcept>

namespace gpu
{
    void pipeline_layout::create(const std::vector<vk::DescriptorSetLayout>& sets,
                                 const std::vector<vk::PushConstantRange>& ranges)
    {
        if(handle_)throw std::logic_error("pipeline layout already created");
        vk::PipelineLayoutCreateInfo info{};
        info.setLayoutCount=std::uint32_t(sets.size());
        info.pSetLayouts=sets.empty()?nullptr:sets.data();
        info.pushConstantRangeCount=std::uint32_t(ranges.size());
        info.pPushConstantRanges=ranges.empty()?nullptr:ranges.data();
        handle_=vk::pipeline_layout::create(info);
    }
    void pipeline_layout::reset()
    {
        if(handle_)vk::pipeline_layout::destroy(handle_);
        handle_=vk::PipelineLayout{};
    }
}
