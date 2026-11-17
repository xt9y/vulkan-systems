#pragma once
#include <array>
#include <chrono>
#include <exception>
#include <cstdint>
#include <vector>
#include <functional>
#include <string>
#include <vk/vk.hpp>
#include <vs/render/frame.hpp>
struct GLFWwindow;
namespace vs { namespace editor { class session; }
// Owns GLFW, VulkanWrapper device, swapchain, and presentation synchronization.
// Individual renderers still own their GPU passes and outputs.
class application {
public:
    struct settings { std::string title="vs"; int width=1280,height=720; std::uint32_t frames_in_flight=2; };
    using frame=::vs::render::frame_scheduler::ticket;
    application()=default;
    explicit application(const settings& options){open(options);}
    ~application(){close();}
    application(const application&)=delete;
    application& operator=(const application&)=delete;
    void open(const settings&);
    bool poll(); // returns false after the window is closed
    frame begin_frame(); // empty command while minimized/recreating swapchain
    void present(frame);
    void resize();
    // Register behavior, not one bool per key or per-frame polling.
    // Key codes use GLFW values; ASCII capitals ('W') work directly.
    void on_press(int key,std::function<void()> action,bool repeat=false);
    void on_release(int key,std::function<void()> action);
    void on_hold(int key,std::function<void(float)> action); // seconds since last poll
    void attach_editor(editor::session& session); // F1, Tab, arrows, mouse, callbacks
    // For user-owned framebuffers/views: release before old swapchain goes away;
    // rebuild after new swapchain images have been installed.
    void on_swapchain(std::function<void()> release,std::function<void()> rebuild);
    // Minimal standalone graphics smoke example, no shader required.
    // A shared color pass for projects that compose their own rendering.
    // Valid while the swapchain is alive; recreate dependents on resize.
    vk::RenderPass color_pass();
    void begin_color(vk::CommandBuffer command,std::uint32_t image,
                     std::array<float,4> rgba={.04f,.05f,.06f,1.f});
    void end_color(vk::CommandBuffer command);
    void clear(vk::CommandBuffer command,std::uint32_t image,
               std::array<float,4> rgba={.04f,.05f,.06f,1.f});
    void run(const std::function<void(vk::CommandBuffer,std::uint32_t)>& draw);
    void close() noexcept;
    GLFWwindow* native_window() const noexcept {return window_;}
    vk::Extent2D extent() const;
    vk::Format format() const;
private:
    struct press_binding {
        int key=0;
        bool repeat=false,editor=false;
        std::function<void()> action;
    };
    struct hold_binding {int key=0;std::function<void(float)> action;};
    std::vector<press_binding> presses_,releases_;
    std::vector<hold_binding> holds_;
    editor::session* editor_=nullptr; // borrowed, must outlive application
    bool editor_cursor_active_=false;
    int cursor_mode_before_editor_=0;
    std::chrono::steady_clock::time_point last_poll_{};
    std::exception_ptr input_error_{};
    void clear_resources() noexcept;
    void create_clear_resources();
    GLFWwindow* window_=nullptr;
    bool glfw_initialized_=false,initialized_=false;
    ::vs::render::frame_scheduler frames_;
    std::function<void()> release_swapchain_,rebuild_swapchain_;
    vk::RenderPass clear_pass_{};
    std::vector<vk::ImageView> clear_views_;
    std::vector<vk::Framebuffer> clear_buffers_;
};
}
