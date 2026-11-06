#pragma once
#include <vk/vk.hpp>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace gpu
{
    // Explicit shader/attachment/transfer accesses; no inferred shader behavior.
    // One GPU queue; image tracking is per (image, exact subresource range).
    struct access
    {
        vk::Buffer buffer{};
        vk::Image image{};
        vk::PipelineStageFlags stage{};
        vk::AccessFlags flags{};
        vk::ImageLayout layout=vk::ImageLayout::eUndefined;
        vk::ImageSubresourceRange range{};
        bool writes=false;

        static access for_buffer(vk::Buffer handle,vk::PipelineStageFlags stage,
                                 vk::AccessFlags flags,bool writes);
        static access for_image(vk::Image handle,vk::ImageSubresourceRange range,
                                vk::ImageLayout layout,vk::PipelineStageFlags stage,
                                vk::AccessFlags flags,bool writes);
    };

    // Reset passes every frame, preserving resource states across ordered submissions.
    // State tracking does not replace semaphores/fences or queue-family ownership transfer.
    class graph
    {
    public:
        using callback=std::function<void(vk::CommandBuffer)>;
        void add(std::string name,std::vector<access> resources,callback record);
        void execute(vk::CommandBuffer command);
        void clear_passes(){passes_.clear();}
        void clear_states(){states_.clear();}
        void import(const access& previous);
    private:
        struct pass
        {
            std::string name;
            std::vector<access> resources;
            callback record;
        };
        struct tracked
        {
            access last;
            bool valid=false;
        };
        static bool same(const access& a,const access& b);
        std::vector<pass> passes_;
        std::vector<tracked> states_;
    };
}
