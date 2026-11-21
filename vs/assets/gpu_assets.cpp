#include <algorithm>
#include <array>
#include <climits>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <vs/assets/gpu_assets.hpp>
#include <vs/assets/texture.hpp>
#include <vs/gpu/upload.hpp>
namespace vs
{
namespace
{
std::uint32_t checksum(const std::uint8_t *bytes, std::size_t size)
{
    std::uint32_t hash = 2166136261u;

    for (std::size_t i = 0; i < size; ++i)
        hash = (hash ^ bytes[i]) * 16777619u;

    return hash;
}
} // namespace
struct gpu_assets::texture_append_state
{
    struct signature
    {
        std::uint32_t id, hash, width, height;
        std::size_t source_index;
    };

    std::vector<signature> textures;
    const ::vs::scene::texture *source_data = nullptr;
    std::size_t source_count = 0;
    std::size_t index = 0;
    std::uint32_t next_row = 0;
    std::uint32_t levels = 1;
    vk::image::resource image{};
};

gpu_assets::gpu_assets(bool anisotropy_enabled)
    : anisotropy_enabled_(anisotropy_enabled) {}
gpu_assets::~gpu_assets() { reset(); }
void gpu_assets::initialize()
{
    if (sampler_) return;

    auto device = vk::device::get()->handle;
    if (!device) throw std::logic_error("vs gpu_assets requires a Vulkan device");
    vk::SamplerCreateInfo info{};
    info.magFilter = info.minFilter = vk::Filter::eLinear;
    info.mipmapMode = vk::SamplerMipmapMode::eLinear;
    info.addressModeU = info.addressModeV = info.addressModeW = vk::SamplerAddressMode::eRepeat;
    info.minLod = 0.f;
    info.maxLod = 32.f;

    if (anisotropy_enabled_ && vk::physical_device::get()->features.samplerAnisotropy) {
        info.anisotropyEnable = VK_TRUE;
        info.maxAnisotropy = std::min(8.f,
                                      vk::physical_device::get()->properties.limits.maxSamplerAnisotropy);
    }

    sampler_ = device.createSampler(info);
    try
    {
        const std::array<std::uint8_t, 4> white{255, 255, 255, 255};
        ::vs::gpu::upload::batch batch;
        cache_.upsert(0, checksum(white.data(), white.size()), white.data(), 1, 1, batch);
        batch.finish();
    }

    catch (...)
    {
        reset();
        throw;
    }
}

void gpu_assets::upload(std::span<const ::vs::scene::mesh> cpu_meshes,
                        std::span<const ::vs::scene::texture> textures)
{
    initialize();
    std::vector<mesh> next;
    next.reserve(cpu_meshes.size());
    ::vs::gpu::upload::batch batch;

    for (const auto &cpu : cpu_meshes)
    {
        if (cpu.vertices.empty() || cpu.indices.empty()) continue;
        auto &target = next.emplace_back();
        target.id = cpu.id;
        target.skin_id = cpu.skin_id;
        target.material = cpu.surface;
        target.geometry.upload(batch, cpu.vertices.data(),
                               cpu.vertices.size() * sizeof(::vs::scene::vertex),
                               cpu.indices.data(), cpu.indices.size() * sizeof(std::uint32_t),
                               std::uint32_t(cpu.indices.size()));
    }

    std::unordered_set<std::uint32_t> live;
    std::vector<std::uint64_t> keep{0};
    keep.reserve(textures.size() + 1);

    for (std::size_t texture_index = 0; texture_index < textures.size(); ++texture_index)
    {
        const auto &tex = textures[texture_index];

        if (tex.id == 0) throw std::invalid_argument("Texture id 0 is reserved for white");
        if (!live.insert(tex.id).second) throw std::invalid_argument("Duplicate texture ID");
        if (!tex.width || !tex.height || tex.width > INT_MAX || tex.height > INT_MAX ||
            std::uint64_t(tex.width) * tex.height * 4u > SIZE_MAX)
            throw std::invalid_argument("Invalid texture extent");

        const auto *bytes = tex.encoded && !tex.encoded->empty()
                                ? tex.encoded.get()
                                : tex.pixels.get();

        if (!bytes) throw std::invalid_argument("Missing texture bytes");
        const std::uint32_t hash = checksum(bytes->data(), bytes->size());

        if (!cache_.contains(tex.id, hash)) {
            std::vector<std::uint8_t> scratch;

            const auto pixels = ::vs::assets::rgba(tex, scratch);
            cache_.upsert(tex.id, hash, pixels.data(), tex.width, tex.height, batch);
        }

        keep.push_back(tex.id);
    }

    batch.finish();
    meshes_ = std::move(next);
    cache_.retain(keep);
    live_ = std::move(live);
}

bool gpu_assets::append_textures(
    std::span<const ::vs::scene::texture> textures)
{
    std::vector<std::uint32_t> hashes;
    hashes.reserve(textures.size());

    for (const auto &tex : textures)
    {
        const auto *bytes = tex.encoded && !tex.encoded->empty()
                                ? tex.encoded.get()
                                : tex.pixels.get();

        if (!bytes) throw std::invalid_argument("Missing texture bytes");
        hashes.push_back(checksum(bytes->data(), bytes->size()));
    }

    return append_textures(textures, hashes);
}

bool gpu_assets::append_textures(
    std::span<const ::vs::scene::texture> textures,
    std::span<const std::uint32_t> hashes)
{
    initialize();

    if (hashes.size() != textures.size())
        throw std::invalid_argument("Texture hash count differs from append batch");
    std::unordered_set<std::uint32_t> added;
    added.reserve(textures.size());
    std::vector<texture_append_state::signature> signatures;
    signatures.reserve(textures.size());

    if (texture_append_) {
        if (textures.data() != texture_append_->source_data ||
            textures.size() != texture_append_->source_count)
            throw std::logic_error("A different texture append is already active");
        signatures = texture_append_->textures;
    }
    else {
        for (std::size_t texture_index = 0; texture_index < textures.size(); ++texture_index)
        {
            const auto &tex = textures[texture_index];

            if (tex.id == 0 || !added.insert(tex.id).second)
                throw std::invalid_argument("Invalid appended texture ID");

            if (!tex.width || !tex.height || tex.width > INT_MAX || tex.height > INT_MAX ||
                std::uint64_t(tex.width) * tex.height * 4u > SIZE_MAX)
                throw std::invalid_argument("Invalid texture extent");
            const std::uint32_t hash = hashes[texture_index];

            if (live_.contains(tex.id)) {
                if (cache_.contains(tex.id, hash)) continue;
                throw std::invalid_argument("Appended texture ID already has different contents");
            }

            signatures.push_back({tex.id, hash, tex.width, tex.height, texture_index});
        }
    }

    if (texture_append_) {
        if (signatures.size() != texture_append_->textures.size())
            throw std::logic_error("A different texture append is already active");

        for (std::size_t i = 0; i < signatures.size(); ++i)
        {
            const auto &a = signatures[i], &b = texture_append_->textures[i];

            if (a.id != b.id || a.hash != b.hash || a.width != b.width || a.height != b.height)
                throw std::logic_error("A different texture append is already active");
        }
    }
    else {
        texture_append_ = std::make_unique<texture_append_state>();
        texture_append_->textures = std::move(signatures);
        texture_append_->source_data = textures.data();
        texture_append_->source_count = textures.size();
    }

    if (!::vs::gpu::upload::ready()) return false;

    auto &state = *texture_append_;

    while (state.index < state.textures.size())
    {
        const auto signature = state.textures[state.index];
        const auto &tex = textures[signature.source_index];
        std::vector<std::uint8_t> scratch;

        const auto pixels = ::vs::assets::rgba(tex, scratch);

        if (cache_.contains(signature.id, signature.hash)) {
            live_.insert(signature.id);
            ++state.index;

            continue;
        }

        if (!state.image.handle) {
            ::vs::gpu::upload::batch allocator;
            state.image = allocator.create_streamed_image(
                signature.width, signature.height, &state.levels);
        }

        const std::size_t row_bytes = std::size_t(signature.width) * 4u;
        constexpr std::size_t byte_budget = 2u * 1024u * 1024u;

        const auto rows = std::max<std::uint32_t>(1u, static_cast<std::uint32_t>(
                                                          std::min<std::size_t>(signature.height - state.next_row,
                                                                                std::max<std::size_t>(1u, byte_budget / row_bytes))));
        const auto offset = std::size_t(state.next_row) * row_bytes;
        ::vs::gpu::upload::batch batch;
        batch.add_image_rows(state.image.handle, pixels.data() + offset,
                             signature.width, signature.height, state.next_row, rows,
                             state.next_row == 0u, state.next_row + rows == signature.height, state.levels);
        batch.finish();
        state.next_row += rows;

        if (state.next_row == signature.height) {
            cache_.adopt(signature.id, signature.hash, signature.width,
                         signature.height, state.levels, state.image);
            state.image = {};
            state.next_row = 0;
            ++state.index;
        }

        return false;
    }

    if (!::vs::gpu::upload::ready()) return false;

    for (const auto &signature : state.textures)
        live_.insert(signature.id);
    texture_append_.reset();

    return true;
}

void gpu_assets::cancel_texture_append()
{
    if (!::vs::gpu::upload::ready())
        throw std::logic_error("Cannot cancel texture append while upload is in flight");
    texture_append_.reset();
}

const gpu_assets::mesh *gpu_assets::find(std::uint32_t id) const noexcept
{
    for (const auto &m : meshes_)
        if (m.id == id) return &m;

    return nullptr;
}

vk::ImageView gpu_assets::texture(std::uint32_t id) const
{
    if (!sampler_) throw std::logic_error("GPU textures are not initialized");

    return cache_.view(live_.contains(id) ? id : 0);
}

void gpu_assets::bind(vk::DescriptorSet set, const ::vs::scene::material &surface,
                      std::uint32_t first_binding) const
{
    if (!sampler_ || !set) throw std::invalid_argument("GPU assets not initialized or null descriptor set");
    std::array<std::uint32_t, 5> ids{surface.base_color_texture, surface.metallic_roughness_texture, surface.normal_texture, surface.occlusion_texture, surface.emissive_texture};
    std::array<vk::DescriptorImageInfo, 5> images{};
    std::array<vk::WriteDescriptorSet, 5> writes{};

    for (std::size_t i = 0; i < ids.size(); ++i)
    {
        images[i].sampler = sampler_;
        images[i].imageView = texture(ids[i]);
        images[i].imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        writes[i].dstSet = set;
        writes[i].dstBinding = first_binding + std::uint32_t(i);
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = vk::DescriptorType::eCombinedImageSampler;
        writes[i].pImageInfo = &images[i];
    }

    vk::device::get()->handle.updateDescriptorSets(writes, {});
}

void gpu_assets::reset() noexcept
{
    texture_append_.reset();
    meshes_.clear();
    live_.clear();
    cache_.clear();

    if (sampler_) {
        try
        {
            vk::device::get()->handle.destroySampler(sampler_);
        }

        catch (...)
        {
        }

        sampler_ = nullptr;
    }
}
} // namespace vs
