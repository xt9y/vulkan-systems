#pragma once
#include <vs/render/virtual.hpp>
#include <vs/render/geometry.hpp>
#include <vs/gpu/residency.hpp>
#include <vs/gpu/retirement.hpp>
#include <vs/gpu/upload.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

namespace vs::render {
// A virtual-geometry page stores a contiguous group of whole meshlets.
// Rendering a partially resident LOD is forbidden: use an entire coarser
// level until every needed page is confirmed uploaded.
struct virtual_page {
    std::uint64_t id=0;
    std::uint32_t level=0,first=0,count=0;
    std::vector<std::uint32_t> indices;
};
struct virtual_pages {
    std::vector<virtual_page> pages;
    std::vector<std::vector<std::uint64_t>> levels;
};
virtual_pages paginate(const virtual_mesh& source,std::uint64_t asset_id,
                       std::size_t target_bytes=64u*1024u);

// Immutable vertex stream belongs to the asset; the page store owns only
// independent index buffers. Upload callbacks remain on the graphics queue.
// The caller must not use a page until the upload submission fence signals.
class virtual_page_pool {
public:
    using page_id=std::uint64_t;
    explicit virtual_page_pool(std::size_t budget_bytes)
        :accounting_(budget_bytes){}
    bool stage(const virtual_page& source, const geometry& vertices,
               ::vs::gpu::upload::batch& transfer,
               std::uint64_t submitted_serial);
    // After ::vs::gpu::upload::ready(), pages become visible to draws.
    void confirm_uploads();
    bool view(page_id id,mesh& out) const;
    void touch(page_id id,std::uint64_t submitted_serial);
    void evict(std::uint64_t completed_serial,
               std::size_t target_bytes=0);
    std::size_t used() const noexcept{return accounting_.used();}
    std::size_t budget() const noexcept{return accounting_.budget();}
    bool ready(page_id id) const;
    // Require every page in an LOD. The base level always remains a fallback
    // when installed; the GPU culler accepts this four-bit residency mask.
    std::uint32_t level_mask(const virtual_pages& catalog) const;
    void clear_idle();
private:
    struct item {
        ::vs::gpu::owned_buffer indices;
        vk::Buffer vertices{};
        std::uint32_t index_count=0;
        bool confirmed=false;
        std::uint64_t last_used=0;
    };
    ::vs::gpu::residency accounting_;
    std::unordered_map<page_id,std::shared_ptr<item>> pages_;
};
} // namespace vs::render
