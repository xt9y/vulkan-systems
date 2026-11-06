#include "renderer.hpp"
#include <stdexcept>
#include <vector>

namespace render
{
    renderer::renderer(vk::CommandBuffer command):command_(command)
    {
        if(!command_)throw std::invalid_argument("null rendering command buffer");
    }

    void renderer::begin_pass(const vk::RenderPassBeginInfo& info)
    {
        if(inside_pass_)throw std::logic_error("render pass already active");
        command_.beginRenderPass(info,vk::SubpassContents::eInline);
        inside_pass_=true;
    }

    void renderer::begin_pass(const target& destination,vk::Rect2D area,
                              const std::vector<vk::ClearValue>& clear_values)
    {
        if(!destination.pass() || !destination.framebuffer())
            throw std::invalid_argument("render target is not initialized");
        vk::RenderPassBeginInfo begin{};
        begin.renderPass=destination.pass();
        begin.framebuffer=destination.framebuffer();
        begin.renderArea=area;
        begin.clearValueCount=std::uint32_t(clear_values.size());
        begin.pClearValues=clear_values.empty()?nullptr:clear_values.data();
        begin_pass(begin);
    }

    void renderer::end_pass()
    {
        if(!inside_pass_)throw std::logic_error("no active render pass");
        command_.endRenderPass();
        inside_pass_=false;
    }

    void renderer::viewport(vk::Viewport view,vk::Rect2D scissor)
    {
        vk::record::viewport(command_,view);
        vk::record::scissor(command_,scissor);
    }

    void renderer::bind(const mesh& geometry,const material& surface) const
    {
        if(!surface.pipeline || !surface.layout)
            throw std::invalid_argument("material requires pipeline and layout");
        vk::record::pipeline(command_,vk::PipelineBindPoint::eGraphics,surface.pipeline);
        if(!surface.sets.empty())
            command_.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                                        surface.layout,0,surface.sets,
                                        surface.dynamic_offsets);
        const auto push_bytes=surface.push_data?surface.push_size:surface.push_constants.size();
        const auto* push=surface.push_data?surface.push_data:surface.push_constants.data();
        if(push_bytes)
        {
            if((push_bytes&3u) || !surface.push_stages || !push ||
               push_bytes>UINT32_MAX)
                throw std::invalid_argument("invalid material push constants");
            command_.pushConstants(surface.layout,surface.push_stages,0,
                                   std::uint32_t(push_bytes),push);
        }
        for(const auto& stream:geometry.streams)
        {
            if(!stream.buffer)throw std::invalid_argument("null vertex buffer");
            command_.bindVertexBuffers(stream.binding,{stream.buffer},{stream.offset});
        }
        if(geometry.index_buffer)
            command_.bindIndexBuffer(geometry.index_buffer,geometry.index_offset,
                                     geometry.index_type);
    }

    void renderer::draw(const mesh& geometry,const material& surface,
                        std::uint32_t instances,std::uint32_t first_instance)
    {
        if(!instances)return;
        bind(geometry,surface);
        if(geometry.index_buffer)
        {
            if(!geometry.index_count)return;
            vk::record::draw_indexed(command_,geometry.index_count,instances,
                                     geometry.first_index,geometry.vertex_offset,
                                     first_instance);
        }
        else
        {
            if(!geometry.vertex_count)return;
            vk::record::draw(command_,geometry.vertex_count,instances,
                             geometry.first_vertex,first_instance);
        }
    }

    void renderer::draw_indirect(const mesh& geometry,const material& surface,
                                 vk::Buffer args,vk::DeviceSize offset,
                                 std::uint32_t count,std::uint32_t stride)
    {
        if(!args || !count || (offset&3u))
            throw std::invalid_argument("invalid indirect draw");
        bind(geometry,surface);
        if(geometry.index_buffer)
        {
            if(!stride)stride=sizeof(vk::DrawIndexedIndirectCommand);
            command_.drawIndexedIndirect(args,offset,count,stride);
        }
        else
        {
            if(!stride)stride=sizeof(vk::DrawIndirectCommand);
            command_.drawIndirect(args,offset,count,stride);
        }
    }
}
