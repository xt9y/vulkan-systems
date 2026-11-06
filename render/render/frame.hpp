#pragma once
#include <vk/vk.hpp>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace render {
// Presentation synchronization for one graphics/present queue pair.
// The caller owns the swapchain and rebuilds it after the device becomes idle.
class frame_scheduler {
public:
    struct ticket {
        vk::CommandBuffer command{};
        std::uint32_t image=0;
    };
    frame_scheduler()=default;
    ~frame_scheduler(){reset();}
    frame_scheduler(const frame_scheduler&)=delete;
    frame_scheduler& operator=(const frame_scheduler&)=delete;

    void create(std::uint32_t frames_in_flight=2);
    // Call only after waiting for GPU idle and recreating the swapchain.
    void resize(std::uint32_t image_count);
    // Wait for the current frame fence before modifying resources it may use.
    void wait();
    // True when every submitted frame has finished.
    bool idle() const;
    // Returns an open command buffer. May throw vk::OutOfDateKHRError.
    ticket begin();
    // Ends, submits, signals image completion and presents.
    // The caller handles eSuboptimalKHR / eErrorOutOfDateKHR and resize.
    vk::Result submit(ticket active);
    void reset() noexcept;
private:
    struct frame {
        vk::CommandBuffer command{};
        vk::Semaphore image_ready{};
        vk::Fence fence{};
    };
    std::vector<frame> frames_;
    std::vector<vk::Fence> image_fences_;
    std::vector<vk::Semaphore> render_ready_;
    std::size_t next_=0;
};
}
