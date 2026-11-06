#pragma once
#include <vk/vk.hpp>
#include <cstdint>
#include <string>
#include <vector>

namespace render
{
    struct shader_stage
    {
        vk::ShaderStageFlagBits stage=vk::ShaderStageFlagBits::eVertex;
        std::string spirv;
        std::string entry="main";
    };

    // Convenience raster pipeline. Complex forward/deferred/multiview and
    // feature-specific pipelines can use create_native instead.
    struct raster_state
    {
        vk::RenderPass pass{};
        vk::PipelineLayout layout{};
        std::uint32_t subpass=0;
        vk::PrimitiveTopology topology=vk::PrimitiveTopology::eTriangleList;
        vk::PolygonMode polygon=vk::PolygonMode::eFill;
        vk::CullModeFlags cull=vk::CullModeFlagBits::eBack;
        vk::FrontFace front=vk::FrontFace::eCounterClockwise;
        vk::SampleCountFlagBits samples=vk::SampleCountFlagBits::e1;
        bool depth_test=true;
        bool depth_write=true;
        bool depth_bias=false;
        bool primitive_restart=false;
        std::vector<vk::VertexInputBindingDescription> bindings;
        std::vector<vk::VertexInputAttributeDescription> attributes;
        std::vector<vk::PipelineColorBlendAttachmentState> blend_attachments;
    };

    class pipeline
    {
    public:
        pipeline()=default;
        ~pipeline(){reset();}
        pipeline(const pipeline&)=delete;
        pipeline& operator=(const pipeline&)=delete;

        void create(const raster_state& state,const std::vector<shader_stage>& stages);
        // Caller controls all Vulkan graphics states and extension structures.
        // The shader stages in info are replaced by stages.
        void create_native(vk::GraphicsPipelineCreateInfo info,
                           const std::vector<shader_stage>& stages);
        void reset();
        vk::Pipeline handle() const{return pipeline_;}
    private:
        vk::Pipeline pipeline_{};
    };
}
