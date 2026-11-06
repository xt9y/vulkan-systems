#include "graph.hpp"
#include <stdexcept>
#include <utility>

namespace gpu
{
    access access::for_buffer(vk::Buffer handle,vk::PipelineStageFlags stage,
                              vk::AccessFlags flags,bool writes)
    {
        access value{};
        value.buffer=handle;
        value.stage=stage;
        value.flags=flags;
        value.writes=writes;
        return value;
    }

    access access::for_image(vk::Image handle,vk::ImageSubresourceRange range,
                             vk::ImageLayout layout,vk::PipelineStageFlags stage,
                             vk::AccessFlags flags,bool writes)
    {
        access value{};
        value.image=handle;
        value.range=range;
        value.layout=layout;
        value.stage=stage;
        value.flags=flags;
        value.writes=writes;
        return value;
    }

    bool graph::same(const access& a,const access& b)
    {
        if(bool(a.image)!=bool(b.image))return false;
        if(a.buffer)return a.buffer==b.buffer;
        return a.image==b.image &&
               a.range.aspectMask==b.range.aspectMask &&
               a.range.baseMipLevel==b.range.baseMipLevel &&
               a.range.levelCount==b.range.levelCount &&
               a.range.baseArrayLayer==b.range.baseArrayLayer &&
               a.range.layerCount==b.range.layerCount;
    }

    void graph::add(std::string name,std::vector<access> uses,callback record)
    {
        if(!record)throw std::invalid_argument("null graph callback");
        for(std::size_t i=0;i<uses.size();++i)
        {
            const auto& u=uses[i];
            if((bool(u.buffer)==bool(u.image)) || !u.stage ||
               (u.image && (!u.range.aspectMask || !u.range.levelCount ||
                            !u.range.layerCount)))
                throw std::invalid_argument("invalid resource access");
            for(std::size_t j=0;j<i;++j)
                if(same(u,uses[j]))
                    throw std::invalid_argument("duplicate pass resource; merge accesses");
        }
        passes_.push_back({std::move(name),std::move(uses),std::move(record)});
    }

    void graph::import(const access& state)
    {
        if(bool(state.buffer)==bool(state.image) || !state.stage)
            throw std::invalid_argument("invalid imported state");
        for(auto& tracked:states_)
            if(same(tracked.last,state))
            {
                tracked.last=state;
                tracked.valid=true;
                return;
            }
        states_.push_back({state,true});
    }

    void graph::execute(vk::CommandBuffer command)
    {
        if(!command)throw std::invalid_argument("null graph command buffer");
        for(const auto& pass:passes_)
        {
            for(const auto& next:pass.resources)
            {
                tracked* state=nullptr;
                for(auto& entry:states_)
                    if(same(entry.last,next)){state=&entry;break;}
                if(!state)
                {
                    states_.push_back({next,false});
                    state=&states_.back();
                }
                const bool initial=!state->valid;
                const auto before=state->last;
                const bool transition=bool(next.image) &&
                    (initial || before.layout!=next.layout);
                const bool hazard=!initial && (before.writes || next.writes);
                if(transition || hazard)
                {
                    const auto src=initial?vk::PipelineStageFlags(
                        vk::PipelineStageFlagBits::eTopOfPipe):before.stage;
                    if(next.buffer)
                    {
                        vk::BufferMemoryBarrier barrier{};
                        barrier.srcAccessMask=initial?vk::AccessFlags{}:before.flags;
                        barrier.dstAccessMask=next.flags;
                        barrier.srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
                        barrier.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
                        barrier.buffer=next.buffer;
                        barrier.offset=0;
                        barrier.size=VK_WHOLE_SIZE;
                        command.pipelineBarrier(src,next.stage,{}, {},{barrier},{});
                    }
                    else
                    {
                        vk::ImageMemoryBarrier barrier{};
                        barrier.srcAccessMask=initial?vk::AccessFlags{}:before.flags;
                        barrier.dstAccessMask=next.flags;
                        barrier.oldLayout=initial?vk::ImageLayout::eUndefined:before.layout;
                        barrier.newLayout=next.layout;
                        barrier.srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
                        barrier.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
                        barrier.image=next.image;
                        barrier.subresourceRange=next.range;
                        command.pipelineBarrier(src,next.stage,{}, {},{}, {barrier});
                    }
                }
                state->last=next;
                state->valid=true;
            }
            pass.record(command);
        }
    }
}
