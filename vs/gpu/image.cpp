#include <vs/gpu/image.hpp>
#include <stdexcept>

namespace vs::gpu
{
    void image::create(const vk::ImageCreateInfo& info,vk::MemoryPropertyFlags memory)
    {
        if(resource_.handle)throw std::logic_error("GPU image already created");
        if(info.extent.width==0 || info.extent.height==0 || info.extent.depth==0)
            throw std::invalid_argument("invalid GPU image extent");
        resource_=vk::image::create(info,memory);
    }

    void image::adopt(vk::image::resource resource)
    {
        if(resource_.handle || !resource.handle)
            throw std::logic_error("invalid GPU image adoption");
        resource_=resource;
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
        resource_.reset();
    }
}
