#include "frame.hpp"
#include <stdexcept>
#include <limits>

namespace render {
void frame_scheduler::create(std::uint32_t count) {
    if(!frames_.empty())throw std::logic_error("frame scheduler already created");
    if(!count)throw std::invalid_argument("frame count must be nonzero");
    auto device=vk::device::get()->handle;
    try {
        auto commands=vk::command::allocate(count);
        frames_.resize(count);
        for(std::size_t i=0;i<count;++i) {
            frames_[i].command=commands[i];
            frames_[i].image_ready=device.createSemaphore({});
            frames_[i].fence=device.createFence(
                vk::FenceCreateInfo(vk::FenceCreateFlagBits::eSignaled));
        }
    }catch(...) { reset(); throw; }
}
void frame_scheduler::resize(std::uint32_t image_count) {
    if(frames_.empty() || !image_count)
        throw std::invalid_argument("invalid swapchain frame count");
    auto device=vk::device::get()->handle;
    for(auto semaphore:render_ready_)if(semaphore)device.destroySemaphore(semaphore);
    render_ready_.clear();
    image_fences_.clear();
    try {
        render_ready_.reserve(image_count);
        for(std::uint32_t i=0;i<image_count;++i)
            render_ready_.push_back(device.createSemaphore({}));
        image_fences_.resize(image_count);
    } catch(...) {
        for(auto semaphore:render_ready_)if(semaphore)device.destroySemaphore(semaphore);
        render_ready_.clear();
        throw;
    }
}
void frame_scheduler::wait() {
    if(frames_.empty())throw std::logic_error("frame scheduler not created");
    if(vk::device::get()->handle.waitForFences(frames_[next_].fence,true,UINT64_MAX)
       !=vk::Result::eSuccess)
        throw std::runtime_error("frame fence wait failed");
}
bool frame_scheduler::idle() const {
    auto device=vk::device::get()->handle;
    for(const auto& frame:frames_)
        if(device.getFenceStatus(frame.fence)!=vk::Result::eSuccess)return false;
    return true;
}
frame_scheduler::ticket frame_scheduler::begin() {
    if(image_fences_.empty())throw std::logic_error("frame scheduler requires resize");
    wait();
    auto device=vk::device::get()->handle;
    auto& current=frames_[next_];
    auto index=vk::swapchain::acquire(UINT64_MAX,current.image_ready).value;
    if(index>=image_fences_.size())throw std::runtime_error("invalid swapchain image index");
    if(image_fences_[index] &&
       device.waitForFences(image_fences_[index],true,UINT64_MAX)!=vk::Result::eSuccess)
        throw std::runtime_error("swapchain image fence wait failed");
    for(auto& fence:image_fences_)
        if(fence==current.fence)fence=vk::Fence{};
    image_fences_[index]=current.fence;
    current.command.reset();
    current.command.begin(vk::CommandBufferBeginInfo{});
    return {current.command,index};
}
vk::Result frame_scheduler::submit(ticket active) {
    if(frames_.empty() || active.image>=render_ready_.size() ||
       active.command!=frames_[next_].command)
        throw std::invalid_argument("stale frame scheduler ticket");
    auto& current=frames_[next_];
    current.command.end();
    vk::device::get()->handle.resetFences(current.fence);
    vk::PipelineStageFlags stage=vk::PipelineStageFlagBits::eColorAttachmentOutput;
    vk::SubmitInfo submit_info{};
    submit_info.waitSemaphoreCount=1;
    submit_info.pWaitSemaphores=&current.image_ready;
    submit_info.pWaitDstStageMask=&stage;
    submit_info.commandBufferCount=1;
    submit_info.pCommandBuffers=&current.command;
    submit_info.signalSemaphoreCount=1;
    submit_info.pSignalSemaphores=&render_ready_[active.image];
    vk::submit::queue(vk::queue::get()->graphics,{submit_info},current.fence);
    vk::PresentInfoKHR info{};
    info.waitSemaphoreCount=1;
    info.pWaitSemaphores=&render_ready_[active.image];
    info.swapchainCount=1;
    info.pSwapchains=&vk::swapchain::get()->handle;
    info.pImageIndices=&active.image;
    next_=(next_+1)%frames_.size();
    return vk::submit::present(vk::queue::get()->present,info);
}
void frame_scheduler::reset() noexcept {
    if(!frames_.empty()||!render_ready_.empty()) {
        try {
            auto device=vk::device::get()->handle;
            if(device) {
                for(auto semaphore:render_ready_)if(semaphore)device.destroySemaphore(semaphore);
                for(const auto& frame:frames_) {
                    if(frame.fence)device.destroyFence(frame.fence);
                    if(frame.image_ready)device.destroySemaphore(frame.image_ready);
                }
            }
        }catch(...) {
            // Device loss: local handles are still invalidated below.
        }
    }
    frames_.clear();
    render_ready_.clear();
    image_fences_.clear();
    next_=0;
}
}
