#include "pipeline.hpp"
#include <array>
#include <stdexcept>

namespace render
{
    void pipeline::create_native(vk::GraphicsPipelineCreateInfo info,
                                 const std::vector<shader_stage>& sources)
    {
        if(pipeline_)throw std::logic_error("graphics pipeline already created");
        if(sources.empty() || !info.layout)
            throw std::invalid_argument("graphics pipeline requires stages and layout");
        std::vector<vk::ShaderModule> modules;
        std::vector<vk::PipelineShaderStageCreateInfo> stages;
        try
        {
            modules.reserve(sources.size());
            stages.reserve(sources.size());
            for(const auto& source:sources)
            {
                if(source.spirv.empty() || source.entry.empty())
                    throw std::invalid_argument("empty graphics shader");
                const auto module=vk::shader::create(vk::shader::read(source.spirv));
                modules.push_back(module);
                vk::PipelineShaderStageCreateInfo stage{};
                stage.stage=source.stage;
                stage.module=module;
                stage.pName=source.entry.c_str();
                stages.push_back(stage);
            }
            info.stageCount=std::uint32_t(stages.size());
            info.pStages=stages.data();
            pipeline_=vk::pipeline::graphics(info);
        }
        catch(...)
        {
            for(auto module:modules)vk::shader::destroy(module);
            throw;
        }
        for(auto module:modules)vk::shader::destroy(module);
    }

    void pipeline::create(const raster_state& state,
                          const std::vector<shader_stage>& stages)
    {
        if(!state.pass || !state.layout)
            throw std::invalid_argument("raster state requires render pass and layout");
        vk::PipelineVertexInputStateCreateInfo vertices{};
        vertices.vertexBindingDescriptionCount=std::uint32_t(state.bindings.size());
        vertices.pVertexBindingDescriptions=state.bindings.data();
        vertices.vertexAttributeDescriptionCount=std::uint32_t(state.attributes.size());
        vertices.pVertexAttributeDescriptions=state.attributes.data();

        vk::PipelineInputAssemblyStateCreateInfo assembly{};
        assembly.topology=state.topology;
        assembly.primitiveRestartEnable=state.primitive_restart;

        // Viewport and scissor are dynamic, avoiding swapchain size coupling.
        vk::PipelineViewportStateCreateInfo viewport{};
        viewport.viewportCount=1;
        viewport.scissorCount=1;
        const std::array<vk::DynamicState,2> dynamic_states{
            vk::DynamicState::eViewport,vk::DynamicState::eScissor};
        vk::PipelineDynamicStateCreateInfo dynamic{};
        dynamic.dynamicStateCount=std::uint32_t(dynamic_states.size());
        dynamic.pDynamicStates=dynamic_states.data();

        vk::PipelineRasterizationStateCreateInfo raster{};
        raster.polygonMode=state.polygon;
        raster.cullMode=state.cull;
        raster.frontFace=state.front;
        raster.depthBiasEnable=state.depth_bias;
        raster.lineWidth=1.f;

        vk::PipelineMultisampleStateCreateInfo sampling{};
        sampling.rasterizationSamples=state.samples;

        vk::PipelineDepthStencilStateCreateInfo depth{};
        depth.depthTestEnable=state.depth_test;
        depth.depthWriteEnable=state.depth_write;
        depth.depthCompareOp=state.depth_compare;

        vk::PipelineColorBlendAttachmentState default_attachment{};
        default_attachment.blendEnable=false;
        default_attachment.colorWriteMask=
            vk::ColorComponentFlagBits::eR |
            vk::ColorComponentFlagBits::eG |
            vk::ColorComponentFlagBits::eB |
            vk::ColorComponentFlagBits::eA;
        const bool has_custom=!state.blend_attachments.empty();
        vk::PipelineColorBlendStateCreateInfo blend{};
        blend.attachmentCount=has_custom?
            std::uint32_t(state.blend_attachments.size()):1u;
        blend.pAttachments=has_custom?state.blend_attachments.data():&default_attachment;

        vk::GraphicsPipelineCreateInfo info{};
        info.pVertexInputState=&vertices;
        info.pInputAssemblyState=&assembly;
        info.pViewportState=&viewport;
        info.pRasterizationState=&raster;
        info.pMultisampleState=&sampling;
        info.pDepthStencilState=&depth;
        info.pColorBlendState=&blend;
        info.pDynamicState=&dynamic;
        info.layout=state.layout;
        info.renderPass=state.pass;
        info.subpass=state.subpass;
        create_native(info,stages);
    }

    void pipeline::reset()
    {
        if(pipeline_)vk::pipeline::destroy(pipeline_);
        pipeline_=vk::Pipeline{};
    }
}
