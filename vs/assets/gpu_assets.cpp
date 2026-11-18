#include <vs/assets/gpu_assets.hpp>
#include <vs/gpu/upload.hpp>
#include <vs/assets/texture.hpp>
#include <algorithm>
#include <array>
#include <climits>
#include <memory>
#include <stdexcept>
namespace vs {
namespace {
std::uint32_t checksum(const std::uint8_t* bytes,std::size_t size) {
    std::uint32_t hash=2166136261u;
    for(std::size_t i=0;i<size;++i)hash=(hash^bytes[i])*16777619u;
    return hash;
}
}
void gpu_assets::initialize() {
    if(sampler_)return;
    auto device=vk::device::get()->handle;
    if(!device)throw std::logic_error("vs gpu_assets requires a Vulkan device");
    vk::SamplerCreateInfo info{};
    info.magFilter=info.minFilter=vk::Filter::eLinear;
    info.mipmapMode=vk::SamplerMipmapMode::eLinear;
    info.addressModeU=info.addressModeV=info.addressModeW=vk::SamplerAddressMode::eRepeat;
    info.minLod=0.f;
    info.maxLod=32.f;
    if(anisotropy_enabled_ && vk::physical_device::get()->features.samplerAnisotropy) {
        info.anisotropyEnable=VK_TRUE;
        info.maxAnisotropy=std::min(8.f,
            vk::physical_device::get()->properties.limits.maxSamplerAnisotropy);
    }
    sampler_=device.createSampler(info);
    try {
        const std::array<std::uint8_t,4> white{255,255,255,255};
        ::vs::gpu::upload::batch batch;
        cache_.upsert(0,checksum(white.data(),white.size()),white.data(),1,1,batch);
        batch.finish();
    }catch(...) {
        reset();
        throw;
    }
}
void gpu_assets::upload(std::span<const ::vs::scene::mesh> cpu_meshes,
                        std::span<const ::vs::scene::texture> textures) {
    initialize();
    std::vector<mesh> next;
    next.reserve(cpu_meshes.size());
    ::vs::gpu::upload::batch batch;
    for(const auto& cpu:cpu_meshes) {
        if(cpu.vertices.empty() || cpu.indices.empty())continue;
        auto& target=next.emplace_back();
        target.id=cpu.id;
        target.skin_id=cpu.skin_id;
        target.material=cpu.surface;
        target.geometry.upload(batch,cpu.vertices.data(),
            cpu.vertices.size()*sizeof(::vs::scene::vertex),
            cpu.indices.data(),cpu.indices.size()*sizeof(std::uint32_t),
            std::uint32_t(cpu.indices.size()));
    }
    std::unordered_set<std::uint32_t> live;
    std::vector<std::uint64_t> keep{0};
    keep.reserve(textures.size()+1);
    for(const auto& tex:textures) {
        if(tex.id==0)throw std::invalid_argument("Texture id 0 is reserved for white");
        if(!live.insert(tex.id).second)throw std::invalid_argument("Duplicate texture ID");
        if(!tex.width || !tex.height || tex.width>INT_MAX || tex.height>INT_MAX ||
           std::uint64_t(tex.width)*tex.height*4u>SIZE_MAX)
            throw std::invalid_argument("Invalid texture extent");
        const auto& bytes=tex.encoded.empty()?tex.pixels:tex.encoded;
        const std::uint32_t hash=checksum(bytes.data(),bytes.size());
        if(!cache_.contains(tex.id,hash)) {
            std::vector<std::uint8_t> scratch;
            const auto pixels=::vs::assets::rgba(tex,scratch);
            cache_.upsert(tex.id,hash,pixels.data(),tex.width,tex.height,batch);
        }
        keep.push_back(tex.id);
    }
    batch.finish();
    meshes_=std::move(next);
    cache_.retain(keep);
    live_=std::move(live);
}
const gpu_assets::mesh* gpu_assets::find(std::uint32_t id) const noexcept {
    for(const auto& m:meshes_)if(m.id==id)return &m;
    return nullptr;
}
vk::ImageView gpu_assets::texture(std::uint32_t id) const {
    if(!sampler_)throw std::logic_error("GPU textures are not initialized");
    return cache_.view(live_.contains(id)?id:0);
}
void gpu_assets::bind(vk::DescriptorSet set,const ::vs::scene::material& surface,
                      std::uint32_t first_binding) const {
    if(!sampler_ || !set)throw std::invalid_argument("GPU assets not initialized or null descriptor set");
    std::array<std::uint32_t,5> ids{surface.base_color_texture,
        surface.metallic_roughness_texture,surface.normal_texture,
        surface.occlusion_texture,surface.emissive_texture};
    std::array<vk::DescriptorImageInfo,5> images{};
    std::array<vk::WriteDescriptorSet,5> writes{};
    for(std::size_t i=0;i<ids.size();++i) {
        images[i].sampler=sampler_;
        images[i].imageView=texture(ids[i]);
        images[i].imageLayout=vk::ImageLayout::eShaderReadOnlyOptimal;
        writes[i].dstSet=set;
        writes[i].dstBinding=first_binding+std::uint32_t(i);
        writes[i].descriptorCount=1;
        writes[i].descriptorType=vk::DescriptorType::eCombinedImageSampler;
        writes[i].pImageInfo=&images[i];
    }
    vk::device::get()->handle.updateDescriptorSets(writes,{});
}
void gpu_assets::reset() noexcept {
    meshes_.clear();
    live_.clear();
    cache_.clear();
    if(sampler_) {
        try {vk::device::get()->handle.destroySampler(sampler_);}catch(...) {}
        sampler_=nullptr;
    }
}
} // namespace vs
