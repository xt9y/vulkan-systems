#pragma once
#include <vk/vk.hpp>
#include <utility>

namespace gpu {
// Move-only RAII resource compatible with VulkanWrapper's resource APIs.
// It is intentionally layout-compatible at the base-class level, so it can
// be passed as vk::buffer::resource& and vk::image::resource& respectively.
// GPU work referencing these resources must finish before the owner resets.
template<class Resource,void(*Destroy)(Resource&)>
class owned_resource:public Resource {
public:
    owned_resource()=default;
    explicit owned_resource(Resource value) noexcept:Resource(value){}
    owned_resource(const owned_resource&)=delete;
    owned_resource& operator=(const owned_resource&)=delete;
    owned_resource(owned_resource&& other) noexcept {
        static_cast<Resource&>(*this)=other.release();
    }
    owned_resource& operator=(owned_resource&& other) noexcept {
        if(this!=&other){reset();static_cast<Resource&>(*this)=other.release();}
        return *this;
    }
    owned_resource& operator=(Resource value) noexcept {
        reset();
        static_cast<Resource&>(*this)=value;
        return *this;
    }
    ~owned_resource(){reset();}
    void reset() noexcept {
        if(this->handle) {
            try {Destroy(*this);}catch(...) {
                // GPU device loss may make explicit destruction unavailable.
                static_cast<Resource&>(*this)=Resource{};
            }
        }
    }
    Resource release() noexcept {
        Resource result=static_cast<Resource&>(*this);
        static_cast<Resource&>(*this)=Resource{};
        return result;
    }
};
using owned_buffer=owned_resource<vk::buffer::resource,vk::buffer::destroy>;
using owned_image=owned_resource<vk::image::resource,vk::image::destroy>;
inline owned_buffer create_buffer(vk::DeviceSize bytes,vk::BufferUsageFlags usage,
                                  vk::MemoryPropertyFlags memory) {
    return owned_buffer(vk::buffer::create(bytes,usage,memory));
}
inline owned_image create_image(const vk::ImageCreateInfo& info,
                                vk::MemoryPropertyFlags memory) {
    return owned_image(vk::image::create(info,memory));
}
}
