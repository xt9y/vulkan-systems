#include <vs/render/virtual_pages.hpp>
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace vs::render {
virtual_pages paginate(const virtual_mesh& source,std::uint64_t asset,
                       std::size_t target_bytes) {
    if(target_bytes<sizeof(std::uint32_t)*3u)
        throw std::invalid_argument("Virtual page size must hold a triangle");
    virtual_pages out;
    out.levels.resize(source.levels.size());
    const auto max_indices=target_bytes/sizeof(std::uint32_t);
    std::uint64_t ordinal=1u;
    for(std::size_t level=0;level<source.levels.size();++level) {
        const auto& lod=source.levels[level];
        if(lod.indices.empty())continue;
        std::uint32_t start=0,count=0;
        const auto flush=[&]() {
            if(!count)return;
            virtual_page page{};
            page.id=(asset<<32u)^ordinal++;
            page.level=std::uint32_t(level);
            page.first=start;
            page.count=count;
            page.indices.assign(lod.indices.begin()+start,
                                lod.indices.begin()+start+count);
            out.levels[level].push_back(page.id);
            out.pages.push_back(std::move(page));
            start+=count;
            count=0;
        };
        for(const auto& cluster:lod.clusters) {
            if(cluster.first!=start+count ||
               cluster.count>max_indices)
                throw std::invalid_argument("Invalid cluster page sequence");
            if(count && std::size_t(count)+cluster.count>max_indices)
                flush();
            count+=cluster.count;
        }
        flush();
        if(start!=lod.indices.size())
            throw std::invalid_argument("Incomplete virtual index hierarchy");
    }
    return out;
}
bool virtual_page_pool::stage(const virtual_page& source,const geometry& owner,
                              ::vs::gpu::upload::batch& transfer,
                              std::uint64_t serial) {
    if(!source.id || !owner.vertex_buffer() || source.indices.empty() ||
       source.indices.size()!=source.count ||
       source.indices.size()>std::numeric_limits<std::uint32_t>::max())
        throw std::invalid_argument("Invalid virtual page");
    if(pages_.contains(source.id)) {
        touch(source.id,serial);
        return true;
    }
    const auto bytes=source.indices.size()*sizeof(std::uint32_t);
    if(!accounting_.insert(source.id,bytes,serial))return false;
    try {
        auto page=std::make_shared<item>();
        page->vertices=owner.vertex_buffer();
        page->index_count=source.count;
        page->last_used=serial;
        page->indices=::vs::gpu::owned_buffer(transfer.add(
            source.indices.data(),bytes,vk::BufferUsageFlagBits::eIndexBuffer));
        pages_.emplace(source.id,std::move(page));
        return true;
    }catch(...) {
        accounting_.remove(source.id,serial);
        throw;
    }
}
void virtual_page_pool::confirm_uploads() {
    // The caller verified completion of the upload queue fence.
    for(auto& [_,item]:pages_)item->confirmed=true;
}
bool virtual_page_pool::ready(page_id id) const {
    const auto it=pages_.find(id);
    return it!=pages_.end() && it->second->confirmed;
}
bool virtual_page_pool::view(page_id id,mesh& out) const {
    const auto it=pages_.find(id);
    if(it==pages_.end() || !it->second->confirmed)return false;
    out={};
    out.streams.push_back({0,it->second->vertices,0});
    out.index_buffer=it->second->indices.handle;
    out.index_type=vk::IndexType::eUint32;
    out.index_count=it->second->index_count;
    return true;
}
void virtual_page_pool::touch(page_id id,std::uint64_t submitted_serial) {
    const auto it=pages_.find(id);
    if(it==pages_.end())return;
    it->second->last_used=std::max(it->second->last_used,submitted_serial);
    accounting_.touch(id,submitted_serial);
}
void virtual_page_pool::evict(std::uint64_t completed_serial,
                             std::size_t target_bytes) {
    for(const auto id:accounting_.trim(completed_serial,target_bytes))
        pages_.erase(id); // index allocation is no longer in flight
}
std::uint32_t virtual_page_pool::level_mask(const virtual_pages& catalog) const {
    std::uint32_t result=0u;
    for(std::size_t level=0;level<catalog.levels.size() && level<4u;++level) {
        const auto& pages=catalog.levels[level];
        if(!pages.empty() && std::all_of(pages.begin(),pages.end(),
            [&](page_id id){return ready(id);}))
            result|=1u<<level;
    }
    return result;
}
void virtual_page_pool::clear_idle() {
    pages_.clear();
    accounting_.clear_idle();
}
} // namespace vs::render
