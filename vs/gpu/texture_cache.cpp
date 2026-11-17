#include <vs/gpu/texture_cache.hpp>
#include <algorithm>
#include <stdexcept>
#include <utility>

namespace vs::gpu {
    bool texture_cache::contains(std::uint64_t key,std::uint32_t hash) const noexcept {
        auto it=images_.find(key);
        return it!=images_.end() && it->second.hash==hash;
    }
    vk::ImageView texture_cache::view(std::uint64_t key) const {
        auto it=images_.find(key);
        if(it==images_.end())throw std::out_of_range("texture not resident");
        return it->second.image.view();
    }
    void texture_cache::upsert(std::uint64_t key,std::uint32_t hash,
        const void* rgba,std::uint32_t width,std::uint32_t height,
        upload::batch& transfers) {
        if(!rgba || !width || !height)throw std::invalid_argument("invalid texture");
        auto it=images_.find(key);
        if(it!=images_.end() && it->second.hash==hash &&
           it->second.width==width && it->second.height==height)return;
        texture incoming{};
        incoming.hash=hash;incoming.width=width;incoming.height=height;
        incoming.image.adopt(transfers.add_image(rgba,width,height));
        vk::ImageViewCreateInfo info{};
        info.viewType=vk::ImageViewType::e2D;
        info.format=incoming.image.format();
        info.subresourceRange={vk::ImageAspectFlagBits::eColor,0,1,0,1};
        incoming.image.make_view(info);
        if(it==images_.end())
            images_.emplace(key,std::move(incoming));
        else
            it->second=std::move(incoming);
    }
    void texture_cache::retain(const std::vector<std::uint64_t>& keys) {
        for(auto it=images_.begin();it!=images_.end();) {
            if(std::find(keys.begin(),keys.end(),it->first)==keys.end()) {
                it=images_.erase(it);
            } else ++it;
        }
    }
    void texture_cache::clear() noexcept {
        images_.clear();
    }
}
