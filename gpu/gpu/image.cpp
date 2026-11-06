#include "image.hpp"
#include <stdexcept>

namespace gpu
{
    void image::create(const vk::ImageCreateInfo& info,vk::MemoryPropertyFlags memory)
    {
        if(resource_.handle)throw std::logic_error("GPU image already created");
        if(info.extent.width==0 || info.extent.height==0 || info.extent.depth==0)
            throw std::invalid_argument("invalid GPU image extent");
        resource_=vk::image::create(info,memory);
    }

    void image::make_view(vk::ImageViewCreateInfo info)
    {
        if(!resource_.handle)throw std::logic_error("GPU image not initialized");
        info.image=resource_.handle;
        if(view_)vk::image_view::destroy(view_);
        view_=vk::ImageView{};
        view_=vk::image_view::create(info);
    }

    void image::reset()
    {
        if(view_)vk::image_view::destroy(view_);
        view_=vk::ImageView{};
        if(resource_.handle)vk::image::destroy(resource_);
        resource_=vk::image::resource{};
    }
}
