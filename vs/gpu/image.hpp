#pragma once
#include <vs/gpu/resource.hpp>
#include <utility>

namespace vs::gpu
{
    class image
    {
    public:
        image()=default;
        ~image(){reset();}
        image(const image&)=delete;
        image& operator=(const image&)=delete;
        image(image&& other) noexcept:
            resource_(std::move(other.resource_)),
            view_(std::exchange(other.view_,vk::ImageView{})){}
        image& operator=(image&& other) noexcept {
            if(this!=&other) {
                reset();
                resource_=std::move(other.resource_);
                view_=std::exchange(other.view_,vk::ImageView{});
            }
            return *this;
        }

        void create(const vk::ImageCreateInfo& info,vk::MemoryPropertyFlags memory);
        void adopt(vk::image::resource resource); // take ownership of staged image
        // Create or replace an owned view. info.image is set automatically.
        void make_view(vk::ImageViewCreateInfo info);
        void reset();
        vk::Image handle() const{return resource_.handle;}
        vk::ImageView view() const{return view_;}
        vk::Format format() const{return resource_.format;}
        vk::Extent3D extent() const{return resource_.size;}
        const vk::image::resource& resource() const{return resource_;}
    private:
        owned_image resource_{};
        vk::ImageView view_{};
    };
}
