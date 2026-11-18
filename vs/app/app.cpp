#include <vs/app/app.hpp>
#include <vs/editor/editor.hpp>
#include <chrono>
#include <exception>
#include <cmath>
#include <GLFW/glfw3.h>
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>
#include <vs/gpu/upload.hpp>
#include <cstdlib>
#include <initializer_list>
#include <string>
#include <vector>
#ifdef __APPLE__
#include <dlfcn.h>
#endif
#include <vk/vk.hpp>

namespace vs::detail::vulkan_loader
{
struct state {
    PFN_vkGetInstanceProcAddr get_instance_proc_addr=nullptr;
    void* library=nullptr;
    bool initialized=false;
};
state* get() {static state data;return &data;}

PFN_vkGetInstanceProcAddr init()
{
    auto& data=*get();
    if(data.initialized)return data.get_instance_proc_addr;
    data.initialized=true;
#ifdef __APPLE__
    std::vector<std::string> paths;
    const auto append=[&](const char* env,std::initializer_list<const char*> suffixes) {
        if(const auto* base=std::getenv(env))
            for(const auto* suffix:suffixes)paths.emplace_back(std::string(base)+suffix);
    };
    append("VULKAN_SDK",{"/lib/libMoltenVK.dylib","/lib/libvulkan.1.dylib"});
    append("HOMEBREW_PREFIX",{"/opt/molten-vk/lib/libMoltenVK.dylib",
                             "/lib/libMoltenVK.dylib","/lib/libvulkan.1.dylib"});
    for(const char* path:{
        "/opt/homebrew/opt/molten-vk/lib/libMoltenVK.dylib",
        "/usr/local/opt/molten-vk/lib/libMoltenVK.dylib",
        "/opt/homebrew/lib/libMoltenVK.dylib",
        "/usr/local/lib/libMoltenVK.dylib","libMoltenVK.dylib",
        "/opt/homebrew/lib/libvulkan.1.dylib",
        "/usr/local/lib/libvulkan.1.dylib","libvulkan.1.dylib"})
        paths.emplace_back(path);
    for(const auto& path:paths) {
        auto* library=dlopen(path.c_str(),RTLD_NOW|RTLD_LOCAL);
        if(!library)continue;
        const auto proc=reinterpret_cast<PFN_vkGetInstanceProcAddr>(
            dlsym(library,"vkGetInstanceProcAddr"));
        if(proc) {
            data.library=library;
            return data.get_instance_proc_addr=proc;
        }
        dlclose(library);
    }
#endif
    return nullptr;
}

void shutdown() noexcept
{
#ifdef __APPLE__
    if(get()->library)dlclose(get()->library);
#endif
    *get()={};
}
} // namespace vs::detail::vulkan_loader

namespace vs {
void application::open(const settings& options) {
    if(glfw_initialized_ || initialized_)throw std::logic_error("vs application already open");
    if(options.width<=0 || options.height<=0 || !options.frames_in_flight)
        throw std::invalid_argument("Invalid window dimensions or frames_in_flight");
    try {
#ifdef __APPLE__
        if(auto loader=detail::vulkan_loader::init())glfwInitVulkanLoader(loader);
#endif
        if(!glfwInit())throw std::runtime_error("GLFW initialization failed");
        glfw_initialized_=true;
        glfwWindowHint(GLFW_CLIENT_API,GLFW_NO_API);
        window_=glfwCreateWindow(options.width,options.height,options.title.c_str(),nullptr,nullptr);
        if(!window_)throw std::runtime_error("Could not create Vulkan window");
        glfwSetWindowUserPointer(window_,this);
        glfwSetKeyCallback(window_,[](GLFWwindow* window,int code,int,int state,int) {
            auto* app=static_cast<application*>(glfwGetWindowUserPointer(window));
            if(!app)return;
            try {
                const auto& bindings=state==GLFW_RELEASE?app->releases_:app->presses_;
                const std::size_t count=bindings.size();
                for(std::size_t i=0;i<count;++i) {
                    if(bindings[i].key!=code ||
                       (state==GLFW_REPEAT && !bindings[i].repeat))continue;
                    if(state!=GLFW_RELEASE && app->editor_ &&
                       app->editor_->captures_input() && !bindings[i].editor)continue;
                    // Callbacks may register more bindings: copy before invocation.
                    auto action=bindings[i].action;
                    action();
                }
            }catch(...) {app->input_error_=std::current_exception();}
        });
        glfwSetCharCallback(window_,[](GLFWwindow* window,unsigned int codepoint) {
            auto* app=static_cast<application*>(glfwGetWindowUserPointer(window));
            if(!app || !app->editor_ || codepoint!=static_cast<unsigned int>('?'))return;
            try {app->editor_->press(editor::key::toggle);}
            catch(...) {app->input_error_=std::current_exception();}
        });
        if(!glfwVulkanSupported())throw std::runtime_error("Vulkan loader/driver unavailable");
#ifdef __APPLE__
        if(!vk::loader::init(detail::vulkan_loader::init()))
#else
        if(!vk::loader::init())
#endif
            throw std::runtime_error("Vulkan loader initialization failed");
        std::uint32_t count=0;
        const char** required=glfwGetRequiredInstanceExtensions(&count);
        if(!required || !count)throw std::runtime_error("Missing Vulkan surface extensions");
        std::vector<const char*> extensions(required,required+count);
        bool portability=false;
        for(const auto& extension:vk::instance::extensions())
            if(std::strcmp(extension.extensionName.data(),VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)==0)
                portability=true;
        if(portability)extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
        vk::ApplicationInfo application{};
        application.pApplicationName=options.title.c_str();
        application.apiVersion=VK_API_VERSION_1_0;
        vk::InstanceCreateInfo info{};
        info.pApplicationInfo=&application;
        info.enabledExtensionCount=std::uint32_t(extensions.size());
        info.ppEnabledExtensionNames=extensions.data();
        if(portability)info.flags=vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR;
        vk::instance::init(info);
        VkSurfaceKHR surface=VK_NULL_HANDLE;
        if(glfwCreateWindowSurface(static_cast<VkInstance>(vk::instance::get()->handle),window_,nullptr,&surface)!=VK_SUCCESS)
            throw std::runtime_error("Could not create Vulkan surface");
        vk::surface::init(vk::SurfaceKHR(surface),true);
        std::uint32_t family=UINT32_MAX;
        for(auto physical:vk::instance::physical_devices()) {
            bool swapchain=false;
            for(const auto& extension:physical.enumerateDeviceExtensionProperties())
                if(std::strcmp(extension.extensionName.data(),VK_KHR_SWAPCHAIN_EXTENSION_NAME)==0)
                    swapchain=true;
            if(!swapchain)continue;
            auto queues=physical.getQueueFamilyProperties();
            for(std::uint32_t i=0;i<queues.size();++i)
                if(queues[i].queueCount && (queues[i].queueFlags & vk::QueueFlagBits::eGraphics) &&
                   (queues[i].queueFlags & vk::QueueFlagBits::eCompute) &&
                   physical.getSurfaceSupportKHR(i,vk::surface::get()->handle)) {
                    vk::physical_device::adopt(physical);
                    family=i;
                    break;
                }
            if(family!=UINT32_MAX)break;
        }
        if(family==UINT32_MAX)throw std::runtime_error("No Vulkan graphics/compute/present device found");
        float priority=1.f;
        vk::DeviceQueueCreateInfo queue{};
        queue.queueFamilyIndex=family;
        queue.queueCount=1;
        queue.pQueuePriorities=&priority;
        std::vector<const char*> device_extensions{VK_KHR_SWAPCHAIN_EXTENSION_NAME};
        if(vk::physical_device::supports_extension("VK_KHR_portability_subset"))
            device_extensions.push_back("VK_KHR_portability_subset");
        vk::DeviceCreateInfo device_info{};
        device_info.queueCreateInfoCount=1;
        device_info.pQueueCreateInfos=&queue;
        device_info.enabledExtensionCount=std::uint32_t(device_extensions.size());
        device_info.ppEnabledExtensionNames=device_extensions.data();
        vk::PhysicalDeviceFeatures enabled{};
        enabled.samplerAnisotropy=vk::physical_device::get()->features.samplerAnisotropy;
        device_info.pEnabledFeatures=&enabled;
        vk::device::init(device_info);
        vk::queue::init(family);
        vk::command::init(family);
        frames_.create(options.frames_in_flight);
        resize();
        initialized_=true;
    } catch(...) {close();throw;}
}
void application::resize() {
    if(!window_)throw std::logic_error("Window not initialized");
    int width=0,height=0;
    glfwGetFramebufferSize(window_,&width,&height);
    if(!width || !height)return;
    vk::device::idle();
    ::vs::gpu::upload::retire(true);
    if(release_swapchain_)release_swapchain_();
    clear_resources();
    auto caps=vk::surface::capabilities();
    if(!(caps.supportedUsageFlags & vk::ImageUsageFlagBits::eColorAttachment))
        throw std::runtime_error("Swapchain does not support color attachment");
    const auto formats=vk::surface::formats();
    auto selected=std::find_if(formats.begin(),formats.end(),[](const vk::SurfaceFormatKHR& v){
        return v.format==vk::Format::eB8G8R8A8Srgb && v.colorSpace==vk::ColorSpaceKHR::eSrgbNonlinear;});
    if(selected==formats.end())
        selected=std::find_if(formats.begin(),formats.end(),[](const vk::SurfaceFormatKHR& v){
            return (v.format==vk::Format::eR8G8B8A8Srgb || v.format==vk::Format::eB8G8R8A8Srgb) &&
                    v.colorSpace==vk::ColorSpaceKHR::eSrgbNonlinear;});
    if(selected==formats.end())throw std::runtime_error("No SRGB swapchain format");
    vk::Extent2D extent=caps.currentExtent;
    if(extent.width==UINT32_MAX) {
        extent.width=std::clamp(std::uint32_t(width),caps.minImageExtent.width,caps.maxImageExtent.width);
        extent.height=std::clamp(std::uint32_t(height),caps.minImageExtent.height,caps.maxImageExtent.height);
    }
    vk::SwapchainCreateInfoKHR info{};
    info.surface=vk::surface::get()->handle;
    info.minImageCount=caps.minImageCount+1;
    if(caps.maxImageCount)info.minImageCount=std::min(info.minImageCount,caps.maxImageCount);
    info.imageFormat=selected->format;
    info.imageColorSpace=selected->colorSpace;
    info.imageExtent=extent;
    info.imageArrayLayers=1;
    info.imageUsage=vk::ImageUsageFlagBits::eColorAttachment;
    info.imageSharingMode=vk::SharingMode::eExclusive;
    info.preTransform=caps.currentTransform;
    info.compositeAlpha=vk::CompositeAlphaFlagBitsKHR::eOpaque;
    info.presentMode=vk::PresentModeKHR::eFifo;
    // FIFO remains the production default. Allow benchmarking WSI latency
    // separately from shader execution without changing the renderer.
    if(const char* requested=std::getenv("VS_PRESENT_MODE")) {
        const auto modes=vk::physical_device::get()->handle
            .getSurfacePresentModesKHR(vk::surface::get()->handle);
        const auto available=[&](vk::PresentModeKHR mode) {
            return std::find(modes.begin(),modes.end(),mode)!=modes.end();
        };
        if(std::strcmp(requested,"mailbox")==0 &&
           available(vk::PresentModeKHR::eMailbox))
            info.presentMode=vk::PresentModeKHR::eMailbox;
        else if(std::strcmp(requested,"immediate")==0 &&
                available(vk::PresentModeKHR::eImmediate))
            info.presentMode=vk::PresentModeKHR::eImmediate;
    }
    info.clipped=true;
    vk::swapchain::init(info);
    frames_.resize(std::uint32_t(vk::swapchain::get()->images.size()));
    if(rebuild_swapchain_)rebuild_swapchain_();
}
void application::on_press(int key,std::function<void()> action,bool repeat)
{
    if(!action)throw std::invalid_argument("Empty key press action");
    presses_.push_back({key,repeat,false,std::move(action)});
}
void application::on_release(int key,std::function<void()> action)
{
    if(!action)throw std::invalid_argument("Empty key release action");
    releases_.push_back({key,false,false,std::move(action)});
}
void application::on_hold(int key,std::function<void(float)> action)
{
    if(!action)throw std::invalid_argument("Empty key hold action");
    holds_.push_back({key,std::move(action)});
}
void application::attach_editor(editor::session& session)
{
    if(!window_ || editor_)throw std::logic_error("An editor is already attached");
    editor_=&session;
    using key=editor::key;
    const auto bind=[&](int code,key command,bool repeat=false) {
        presses_.push_back({code,repeat,true,[this,command] {
            const bool shift=glfwGetKey(window_,GLFW_KEY_LEFT_SHIFT)==GLFW_PRESS ||
                             glfwGetKey(window_,GLFW_KEY_RIGHT_SHIFT)==GLFW_PRESS;
            const bool accelerated=glfwGetKey(window_,GLFW_KEY_LEFT_SUPER)==GLFW_PRESS ||
                                   glfwGetKey(window_,GLFW_KEY_RIGHT_SUPER)==GLFW_PRESS ||
                                   glfwGetKey(window_,GLFW_KEY_LEFT_CONTROL)==GLFW_PRESS ||
                                   glfwGetKey(window_,GLFW_KEY_RIGHT_CONTROL)==GLFW_PRESS;
            editor_->press(command,shift,accelerated);
        }});
    };
    bind(GLFW_KEY_F1,key::toggle);
    bind(GLFW_KEY_TAB,key::cursor);
    bind(GLFW_KEY_ENTER,key::confirm);
    bind(GLFW_KEY_KP_ENTER,key::confirm);
    bind(GLFW_KEY_ESCAPE,key::cancel);
    bind(GLFW_KEY_UP,key::up,true);
    bind(GLFW_KEY_DOWN,key::down,true);
    bind(GLFW_KEY_LEFT,key::left,true);
    bind(GLFW_KEY_RIGHT,key::right,true);
    bind(GLFW_KEY_DELETE,key::erase);
    for(int i=0;i<9;++i)
        bind(GLFW_KEY_1+i,static_cast<key>(int(key::view_1)+i));
}
bool application::poll() {
    if(!window_)return false;
    glfwPollEvents();
    if(input_error_) {
        auto error=std::exchange(input_error_,std::exception_ptr{});
        std::rethrow_exception(error);
    }
    const auto now=std::chrono::steady_clock::now();
    const float dt=last_poll_==std::chrono::steady_clock::time_point{} ?
        0.f:std::clamp(std::chrono::duration<float>(now-last_poll_).count(),0.f,.1f);
    last_poll_=now;
    if(editor_) {
        const bool need_cursor=editor_->visible() && editor_->cursor_free();
        if(need_cursor && !editor_cursor_active_) {
            cursor_mode_before_editor_=glfwGetInputMode(window_,GLFW_CURSOR);
            glfwSetInputMode(window_,GLFW_CURSOR,GLFW_CURSOR_NORMAL);
            editor_cursor_active_=true;
        }else if(!need_cursor && editor_cursor_active_) {
            glfwSetInputMode(window_,GLFW_CURSOR,cursor_mode_before_editor_);
            editor_cursor_active_=false;
        }
        double x=0,y=0;
        int width=0,height=0,fb_width=0,fb_height=0;
        glfwGetCursorPos(window_,&x,&y);
        glfwGetWindowSize(window_,&width,&height);
        glfwGetFramebufferSize(window_,&fb_width,&fb_height);
        const int pixel_x=width>0?int(x*fb_width/width):-1;
        const int pixel_y=height>0?int(y*fb_height/height):-1;
        editor_->pointer(pixel_x,pixel_y,fb_width,fb_height,
                         glfwGetMouseButton(window_,GLFW_MOUSE_BUTTON_LEFT)==GLFW_PRESS);
    }
    if(!editor_ || !editor_->captures_input()) {
        const std::size_t count=holds_.size();
        for(std::size_t i=0;i<count;++i)
            if(glfwGetKey(window_,holds_[i].key)==GLFW_PRESS) {
                auto action=holds_[i].action;
                action(dt);
            }
    }
    if(editor_)editor_->dispatch();
    return !glfwWindowShouldClose(window_);
}
application::frame application::begin_frame() {
    if(!initialized_)throw std::logic_error("Application not initialized");
    int w=0,h=0;
    glfwGetFramebufferSize(window_,&w,&h);
    if(!w || !h)return {};
    const auto caps=vk::surface::capabilities();
    auto extent=caps.currentExtent;
    if(extent.width==UINT32_MAX) {
        extent.width=std::clamp(std::uint32_t(w),caps.minImageExtent.width,caps.maxImageExtent.width);
        extent.height=std::clamp(std::uint32_t(h),caps.minImageExtent.height,caps.maxImageExtent.height);
    }
    if(extent!=vk::swapchain::get()->extent){resize();return {};}
    try {return frames_.begin();}
    catch(const vk::OutOfDateKHRError&) {resize();return {};}
}
bool application::present(frame ticket) {
    if(!ticket.command)return false;
    try {
        auto result=frames_.submit(ticket);
        if(result==vk::Result::eSuboptimalKHR ||
           result==vk::Result::eErrorOutOfDateKHR) {
            resize();
            return false;
        }
        return true;
    } catch(const vk::OutOfDateKHRError&) {
        resize();
        return false;
    }
}
void application::on_swapchain(std::function<void()> release,
                              std::function<void()> rebuild) {
    if(release_swapchain_ || rebuild_swapchain_)
        throw std::logic_error("Swapchain callbacks already registered");
    release_swapchain_=std::move(release);
    rebuild_swapchain_=std::move(rebuild);
    if(initialized_ && rebuild_swapchain_)rebuild_swapchain_();
}
void application::clear_resources() noexcept {
    if(!clear_pass_ && clear_views_.empty() && clear_buffers_.empty())return;
    try {
        auto device=vk::device::get()->handle;
        if(device) {
            for(auto fb:clear_buffers_)if(fb)device.destroyFramebuffer(fb);
            for(auto view:clear_views_)if(view)device.destroyImageView(view);
            if(clear_pass_)device.destroyRenderPass(clear_pass_);
        }
    }catch(...) {}
    clear_buffers_.clear();
    clear_views_.clear();
    clear_pass_=nullptr;
}
void application::create_clear_resources() {
    if(clear_pass_)return;
    auto device=vk::device::get()->handle;
    vk::AttachmentDescription attachment{};
    attachment.format=format();
    attachment.samples=vk::SampleCountFlagBits::e1;
    attachment.loadOp=vk::AttachmentLoadOp::eClear;
    attachment.storeOp=vk::AttachmentStoreOp::eStore;
    attachment.stencilLoadOp=vk::AttachmentLoadOp::eDontCare;
    attachment.stencilStoreOp=vk::AttachmentStoreOp::eDontCare;
    attachment.initialLayout=vk::ImageLayout::eUndefined;
    attachment.finalLayout=vk::ImageLayout::ePresentSrcKHR;
    vk::AttachmentReference reference{0,vk::ImageLayout::eColorAttachmentOptimal};
    vk::SubpassDescription subpass{};
    subpass.pipelineBindPoint=vk::PipelineBindPoint::eGraphics;
    subpass.colorAttachmentCount=1;
    subpass.pColorAttachments=&reference;
    vk::SubpassDependency dependency{};
    dependency.srcSubpass=VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass=0;
    dependency.srcStageMask=vk::PipelineStageFlagBits::eColorAttachmentOutput;
    dependency.dstStageMask=vk::PipelineStageFlagBits::eColorAttachmentOutput;
    dependency.dstAccessMask=vk::AccessFlagBits::eColorAttachmentWrite;
    vk::RenderPassCreateInfo info{};
    info.attachmentCount=1;
    info.pAttachments=&attachment;
    info.subpassCount=1;
    info.pSubpasses=&subpass;
    info.dependencyCount=1;
    info.pDependencies=&dependency;
    try {
        clear_pass_=device.createRenderPass(info);
        for(auto image:vk::swapchain::get()->images) {
            vk::ImageViewCreateInfo view_info{};
            view_info.image=image;
            view_info.viewType=vk::ImageViewType::e2D;
            view_info.format=format();
            view_info.subresourceRange=vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor,0,1,0,1};
            auto view=device.createImageView(view_info);
            clear_views_.push_back(view);
            vk::FramebufferCreateInfo framebuffer{};
            framebuffer.renderPass=clear_pass_;
            framebuffer.attachmentCount=1;
            framebuffer.pAttachments=&clear_views_.back();
            framebuffer.width=extent().width;
            framebuffer.height=extent().height;
            framebuffer.layers=1;
            clear_buffers_.push_back(device.createFramebuffer(framebuffer));
        }
    }catch(...) {clear_resources();throw;}
}
vk::RenderPass application::color_pass() {
    if(!initialized_)throw std::logic_error("Application not initialized");
    create_clear_resources();
    return clear_pass_;
}
void application::begin_color(vk::CommandBuffer command,std::uint32_t image,
                              std::array<float,4> rgba) {
    if(!initialized_ || !command)throw std::logic_error("No active VS frame");
    create_clear_resources();
    if(image>=clear_buffers_.size())throw std::out_of_range("Swapchain image index");
    vk::ClearValue value{};
    value.color.float32=rgba;
    vk::RenderPassBeginInfo info{};
    info.renderPass=clear_pass_;
    info.framebuffer=clear_buffers_[image];
    vk::Rect2D area{};
    area.offset=vk::Offset2D{0,0};
    area.extent=extent();
    info.renderArea=area;
    info.clearValueCount=1;
    info.pClearValues=&value;
    command.beginRenderPass(info,vk::SubpassContents::eInline);
}
void application::end_color(vk::CommandBuffer command) {
    command.endRenderPass();
}
void application::clear(vk::CommandBuffer command,std::uint32_t image,
                        std::array<float,4> rgba) {
    begin_color(command,image,rgba);
    end_color(command);
}
void application::run(const std::function<void(vk::CommandBuffer,std::uint32_t)>& draw) {
    while(poll()) {
        auto ticket=begin_frame();
        if(ticket.command) {draw(ticket.command,ticket.image);present(ticket);}
    }
}
vk::Extent2D application::extent() const {
    return initialized_?vk::swapchain::get()->extent:vk::Extent2D{};
}
vk::Format application::format() const {
    return initialized_?vk::swapchain::get()->format:vk::Format::eUndefined;
}
void application::close() noexcept {
    if(glfw_initialized_) {
        try {
            if(vk::device::get()->handle) {
                vk::device::idle();
                ::vs::gpu::upload::retire(true);
                if(release_swapchain_)release_swapchain_();
                clear_resources();
            }
            frames_.reset();
            vk::command::shutdown();
            vk::swapchain::shutdown();
            vk::queue::shutdown();
            vk::device::shutdown();
            vk::physical_device::shutdown();
            vk::surface::shutdown();
            vk::instance::shutdown();
            vk::loader::shutdown();
        } catch(...) {}
        if(window_)glfwDestroyWindow(window_);
        glfwTerminate();
    }
    detail::vulkan_loader::shutdown();
    window_=nullptr;
    glfw_initialized_=initialized_=false;
    release_swapchain_={};
    rebuild_swapchain_={};
    presses_.clear();releases_.clear();holds_.clear();
    editor_=nullptr;
    editor_cursor_active_=false;
    cursor_mode_before_editor_=0;
    last_poll_={};
    input_error_={};
}
} // namespace vs
