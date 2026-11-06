#pragma once
#include <vk/vk.hpp>

namespace gpu
{
    class image
    {
    public:
        image()=default;
        ~image(){reset();}
        image(const image&)=delete;
        image& operator=(const image&)=delete;

        void create(const vk::ImageCreateInfo& info,vk::MemoryPropertyFlags memory);
        // Create or replace an owned view. info.image is set automatically.
        void make_view(vk::ImageViewCreateInfo info);
        void reset();
        vk::Image handle() const{return resource_.handle;}
        vk::ImageView view() const{return view_;}
        vk::Format format() const{return resource_.format;}
        vk::Extent3D extent() const{return resource_.size;}
        const vk::image::resource& resource() const{return resource_;}
    private:
        vk::image::resource resource_{};
        vk::ImageView view_{};
    };
}
