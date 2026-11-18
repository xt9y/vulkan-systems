#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <vector>

namespace vs::gpu {
// Resource budget/lifetime policy independent of texture, mesh or chunk type.
// Eviction is permitted only after all referencing GPU submissions complete.
// Callers perform Vulkan destruction for returned victims.
class residency {
public:
    using page_id=std::uint64_t;
    struct page {
        std::size_t bytes=0;
        std::uint64_t last_used=0;
        bool pinned=false;
    };
    explicit residency(std::size_t budget_bytes)
        :budget_(budget_bytes){}
    bool insert(page_id id,std::size_t bytes,std::uint64_t submitted,
                bool pinned=false) {
        if(!bytes || bytes>budget_ || pages_.contains(id) ||
           bytes>std::numeric_limits<std::size_t>::max()-used_)
            return false;
        pages_.emplace(id,page{bytes,submitted,pinned});
        used_+=bytes;
        return true;
    }
    bool touch(page_id id,std::uint64_t submitted) {
        const auto it=pages_.find(id);
        if(it==pages_.end())return false;
        it->second.last_used=std::max(it->second.last_used,submitted);
        return true;
    }
    bool pin(page_id id,bool value) {
        const auto it=pages_.find(id);
        if(it==pages_.end())return false;
        it->second.pinned=value;
        return true;
    }
    std::vector<page_id> trim(std::uint64_t completed,
                             std::size_t target_bytes=0) {
        // A zero target means the configured residency budget.
        const auto target=target_bytes?std::min(target_bytes,budget_):budget_;
        std::vector<page_id> candidates;
        for(const auto& [id,page]:pages_)
            if(!page.pinned && page.last_used<=completed)
                candidates.push_back(id);
        std::sort(candidates.begin(),candidates.end(),[&](auto a,auto b){
            const auto& x=pages_.at(a);
            const auto& y=pages_.at(b);
            return x.last_used!=y.last_used?x.last_used<y.last_used:a<b;
        });
        std::vector<page_id> victims;
        for(const auto id:candidates) {
            if(used_<=target)break;
            used_-=pages_.at(id).bytes;
            pages_.erase(id);
            victims.push_back(id);
        }
        return victims;
    }
    bool remove(page_id id,std::uint64_t completed) {
        const auto it=pages_.find(id);
        if(it==pages_.end() || it->second.last_used>completed)return false;
        used_-=it->second.bytes;pages_.erase(it);
        return true;
    }
    const page* find(page_id id) const {
        const auto it=pages_.find(id);
        return it==pages_.end()?nullptr:&it->second;
    }
    void clear_idle(){pages_.clear();used_=0;}
    std::size_t used() const noexcept{return used_;}
    std::size_t budget() const noexcept{return budget_;}
    std::size_t size() const noexcept{return pages_.size();}
private:
    std::size_t budget_=0,used_=0;
    std::unordered_map<page_id,page> pages_;
};
} // namespace vs::gpu
