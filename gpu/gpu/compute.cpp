#include "compute.hpp"
#include <stdexcept>

namespace gpu
{
    void compute::create(const std::string& path,
                         const std::vector<vk::DescriptorSetLayout>& layouts,
                         std::uint32_t push_bytes,const char* entry)
    {
        if(pipeline_ || layout_)throw std::logic_error("compute already created");
        if(path.empty() || !entry || !*entry || (push_bytes&3u))
            throw std::invalid_argument("invalid compute pipeline description");
        vk::ShaderModule shader{};
        try
        {
            vk::PushConstantRange range{};
            if(push_bytes)
            {
                range.stageFlags=vk::ShaderStageFlagBits::eCompute;
                range.size=push_bytes;
            }
            vk::PipelineLayoutCreateInfo info{};
            info.setLayoutCount=std::uint32_t(layouts.size());
            info.pSetLayouts=layouts.empty()?nullptr:layouts.data();
            info.pushConstantRangeCount=push_bytes?1u:0u;
            info.pPushConstantRanges=push_bytes?&range:nullptr;
            layout_=vk::pipeline_layout::create(info);
            shader=vk::shader::create(vk::shader::read(path));
            vk::ComputePipelineCreateInfo pipeline{};
            pipeline.stage.stage=vk::ShaderStageFlagBits::eCompute;
            pipeline.stage.module=shader;
            pipeline.stage.pName=entry;
            pipeline.layout=layout_;
            pipeline_=vk::pipeline::compute(pipeline);
            push_limit_=push_bytes;
            descriptor_count_=std::uint32_t(layouts.size());
            vk::shader::destroy(shader);
        }
        catch(...)
        {
            if(shader)vk::shader::destroy(shader);
            reset();
            throw;
        }
    }

    void compute::reset()
    {
        if(pipeline_)vk::pipeline::destroy(pipeline_);
        pipeline_=vk::Pipeline{};
        if(layout_)vk::pipeline_layout::destroy(layout_);
        layout_=vk::PipelineLayout{};
        push_limit_=descriptor_count_=0;
    }

    void compute::bind(vk::CommandBuffer command,
                       const std::vector<vk::DescriptorSet>& sets,
                       const void* push,std::uint32_t push_bytes) const
    {
        if(!command || !pipeline_ || sets.size()!=descriptor_count_ ||
           push_bytes>push_limit_ || (push_bytes && (!push || (push_bytes&3u))))
            throw std::invalid_argument("invalid compute dispatch");
        vk::record::pipeline(command,vk::PipelineBindPoint::eCompute,pipeline_);
        if(!sets.empty())
            command.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                       layout_,0,sets,{});
        if(push_bytes)
            command.pushConstants(layout_,vk::ShaderStageFlagBits::eCompute,
                                  0,push_bytes,push);
    }

    void compute::dispatch(vk::CommandBuffer command,
                           std::uint32_t x,std::uint32_t y,std::uint32_t z,
                           const std::vector<vk::DescriptorSet>& sets,
                           const void* push,std::uint32_t push_bytes) const
    {
        if(!x || !y || !z)throw std::invalid_argument("zero compute groups");
        bind(command,sets,push,push_bytes);
        vk::record::dispatch(command,x,y,z);
    }

    void compute::dispatch_indirect(vk::CommandBuffer command,vk::Buffer arguments,
                                    vk::DeviceSize offset,
                                    const std::vector<vk::DescriptorSet>& sets,
                                    const void* push,std::uint32_t push_bytes) const
    {
        if(!arguments || (offset&3u))
            throw std::invalid_argument("invalid compute indirect arguments");
        bind(command,sets,push,push_bytes);
        command.dispatchIndirect(arguments,offset);
    }
}
