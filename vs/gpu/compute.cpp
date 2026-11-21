#include <stdexcept>
#include <vs/gpu/compute.hpp>
#include <vs/gpu/shader.hpp>

namespace vs::gpu
{
void compute::create(const std::string &path,
                     const std::vector<vk::DescriptorSetLayout> &layouts,
                     std::uint32_t push_bytes, const char *entry)
{
    if (pipeline_ || layout_.handle()) throw std::logic_error("compute already created");
    if (path.empty() || !entry || !*entry || (push_bytes & 3u))
        throw std::invalid_argument("invalid compute pipeline description");
    try
    {
        vk::PushConstantRange range{};
        std::vector<vk::PushConstantRange> ranges;

        if (push_bytes) {
            range.stageFlags = vk::ShaderStageFlagBits::eCompute;
            range.size = push_bytes;
            ranges.push_back(range);
        }

        layout_.create(layouts, ranges);

        shader module;
        module.load(path);
        vk::ComputePipelineCreateInfo pipeline{};
        pipeline.stage.stage = vk::ShaderStageFlagBits::eCompute;
        pipeline.stage.module = module.handle();
        pipeline.stage.pName = entry;
        pipeline.layout = layout_.handle();
        pipeline_ = vk::pipeline::compute(pipeline);
        push_limit_ = push_bytes;
        descriptor_count_ = std::uint32_t(layouts.size());
    }

    catch (...)
    {
        reset();
        throw;
    }
}

void compute::reset()
{
    if (pipeline_) vk::pipeline::destroy(pipeline_);
    pipeline_ = vk::Pipeline{};
    layout_.reset();
    push_limit_ = descriptor_count_ = 0;
}

void compute::bind(vk::CommandBuffer command,
                   const std::vector<vk::DescriptorSet> &sets,
                   const void *push, std::uint32_t push_bytes) const
{
    if (!command || !pipeline_ || sets.size() != descriptor_count_ ||
        push_bytes > push_limit_ || (push_bytes && (!push || (push_bytes & 3u))))
        throw std::invalid_argument("invalid compute dispatch");
    vk::record::pipeline(command, vk::PipelineBindPoint::eCompute, pipeline_);

    if (!sets.empty())
        command.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                   layout_.handle(), 0, sets, {});
    if (push_bytes)
        command.pushConstants(layout_.handle(), vk::ShaderStageFlagBits::eCompute,
                              0, push_bytes, push);
}

void compute::dispatch(vk::CommandBuffer command,
                       std::uint32_t x, std::uint32_t y, std::uint32_t z,
                       const std::vector<vk::DescriptorSet> &sets,
                       const void *push, std::uint32_t push_bytes) const
{
    if (!x || !y || !z) throw std::invalid_argument("zero compute groups");
    bind(command, sets, push, push_bytes);
    vk::record::dispatch(command, x, y, z);
}

void compute::dispatch_indirect(vk::CommandBuffer command, vk::Buffer arguments,
                                vk::DeviceSize offset,
                                const std::vector<vk::DescriptorSet> &sets,
                                const void *push, std::uint32_t push_bytes) const
{
    if (!arguments || (offset & 3u))
        throw std::invalid_argument("invalid compute indirect arguments");
    bind(command, sets, push, push_bytes);
    command.dispatchIndirect(arguments, offset);
}
} // namespace vs::gpu
