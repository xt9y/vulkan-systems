#include <vs/bvh/gpu_regions.hpp>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace vs::bvh {
namespace {
bool before(spatial::region_id a,spatial::region_id b) {
    if(a.y!=b.y)return a.y<b.y;
    if(a.z!=b.z)return a.z<b.z;
    return a.x<b.x;
}
}
void gpu_regions::create(std::uint32_t frames,std::size_t max_regions,
                         std::size_t max_vertices,std::size_t max_nodes) {
    if(!frames || !max_regions || !max_vertices || !max_nodes ||
       !frames_.empty() || max_regions>UINT32_MAX-1u ||
       max_vertices>UINT32_MAX || max_nodes>UINT32_MAX)
        throw std::invalid_argument("Invalid GPU region atlas capacity");
    const auto storage=vk::BufferUsageFlagBits::eStorageBuffer;
    const auto memory=vk::MemoryPropertyFlagBits::eHostVisible |
                      vk::MemoryPropertyFlagBits::eHostCoherent;
    try {
        region_capacity_=max_regions;
        vertex_capacity_=max_vertices;
        node_capacity_=max_nodes;
        frames_.reserve(frames);
        for(std::uint32_t i=0;i<frames;++i) {
            auto page=std::make_unique<frame>();
            page->records=::vs::gpu::create_buffer(
                (max_regions+1u)*sizeof(record),storage,memory);
            page->vertices=::vs::gpu::create_buffer(
                max_vertices*sizeof(vertex),storage,memory);
            page->nodes=::vs::gpu::create_buffer(
                max_nodes*sizeof(stackless_node),storage,memory);
            frames_.push_back(std::move(page));
            upload(i);
        }
    } catch(...) {reset();throw;}
}
bool gpu_regions::replace(spatial::region_id id,std::uint64_t revision,
                          std::vector<vertex> vertices) {
    if(!revision || vertices.size()%3u)return false;
    const auto known=versions_.find(id);
    if(known!=versions_.end() && revision<=known->second)return false;
    entry next{};
    next.revision=revision;
    next.nodes=build_stackless(vertices);
    store_triangle_edges(vertices);
    next.vertices=std::move(vertices);
    std::size_t count_vertices=next.vertices.size();
    std::size_t count_nodes=next.nodes.size();
    for(const auto& [key,old]:entries_) {
        if(key==id)continue;
        count_vertices+=old.vertices.size();
        count_nodes+=old.nodes.size();
    }
    if(count_vertices>vertex_capacity_ || count_nodes>node_capacity_ ||
       entries_.size()+(entries_.contains(id)?0u:1u)>region_capacity_)
        throw std::length_error("Static shadow region atlas is full");
    entries_.insert_or_assign(id,std::move(next));
    versions_[id]=revision;
    ++serial_;
    return true;
}
bool gpu_regions::replace(spatial::region_id id,std::uint64_t revision,
                          std::span<const scene::triangle> faces) {
    std::vector<vertex> packed;
    packed.reserve(faces.size()*3u);
    for(const auto& face:faces) {
        packed.push_back({face.a.x,face.a.y,face.a.z,0.f});
        packed.push_back({face.b.x,face.b.y,face.b.z,0.f});
        packed.push_back({face.c.x,face.c.y,face.c.z,0.f});
    }
    return replace(id,revision,std::move(packed));
}
bool gpu_regions::erase(spatial::region_id id,std::uint64_t revision) {
    if(!revision)return false;
    const auto known=versions_.find(id);
    if(known!=versions_.end() && revision<=known->second)return false;
    entries_.erase(id);
    versions_[id]=revision;
    ++serial_;
    return true;
}
bool gpu_regions::upload(std::uint32_t image) {
    auto& dest=*frames_.at(image);
    if(dest.serial==serial_)return false;
    std::vector<spatial::region_id> ids;
    ids.reserve(entries_.size());
    for(const auto& [id,_]:entries_)ids.push_back(id);
    std::sort(ids.begin(),ids.end(),before);
    std::vector<record> records(ids.size()+1u);
    records[0].span[0]=std::uint32_t(ids.size());
    std::unordered_map<spatial::region_id,location,spatial::region_hash> next;
    std::size_t vertex_at=0,node_at=0;
    for(std::size_t i=0;i<ids.size();++i) {
        const auto& entry=entries_.at(ids[i]);
        const location where{entry.revision,
            std::uint32_t(vertex_at),std::uint32_t(node_at)};
        auto& item=records[i+1u];
        if(!entry.nodes.empty()) {
            const auto& root=entry.nodes.front();
            item.minimum={root.minimum[0],root.minimum[1],root.minimum[2],0.f};
            item.maximum={root.maximum[0],root.maximum[1],root.maximum[2],0.f};
        }
        item.span={where.vertex_offset,where.node_offset,
            std::uint32_t(entry.nodes.size()),
            std::uint32_t(entry.vertices.size())};
        const auto old=dest.resident.find(ids[i]);
        if(old==dest.resident.end() || !(old->second==where)) {
            if(!entry.vertices.empty())
                vk::buffer::write(dest.vertices,entry.vertices.data(),
                    entry.vertices.size()*sizeof(vertex),
                    vertex_at*sizeof(vertex));
            if(!entry.nodes.empty())
                vk::buffer::write(dest.nodes,entry.nodes.data(),
                    entry.nodes.size()*sizeof(stackless_node),
                    node_at*sizeof(stackless_node));
        }
        next.emplace(ids[i],where);
        vertex_at+=entry.vertices.size();
        node_at+=entry.nodes.size();
    }
    vk::buffer::write(dest.records,records.data(),
                      records.size()*sizeof(record));
    dest.resident=std::move(next);
    dest.serial=serial_;
    return true;
}
gpu_regions::bindings gpu_regions::descriptors(std::uint32_t image) const {
    const auto& frame=*frames_.at(image);
    return {vk::buffer::descriptor(frame.records),
            vk::buffer::descriptor(frame.vertices),
            vk::buffer::descriptor(frame.nodes)};
}
void gpu_regions::reset() noexcept {
    frames_.clear();
    entries_.clear();
    versions_.clear();
    region_capacity_=vertex_capacity_=node_capacity_=0;
    serial_=0;
}
} // namespace vs::bvh
