#pragma once
#include <vs/bvh/stackless.hpp>
#include <vs/spatial/world.hpp>
#include <array>
#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

namespace vs::bvh {
// Independent static acceleration structures per spatial region. Rebuilding
// one region leaves every other CPU BVH untouched. GPU residency is separate.
class region_trees {
public:
    using triangle_vertices=std::vector<std::array<float,4>>;
    struct tree {
        std::uint64_t revision=0;
        triangle_vertices vertices;
        std::vector<stackless_node> nodes;
    };
    bool replace(::vs::spatial::region_id id,std::uint64_t revision,
                 triangle_vertices triangles) {
        if(!revision || triangles.size()%3u)
            return false;
        const auto it=trees_.find(id);
        if(it!=trees_.end() && revision<=it->second.revision)return false;
        tree next;
        next.revision=revision;
        next.nodes=build_stackless(triangles);
        store_triangle_edges(triangles);
        next.vertices=std::move(triangles);
        trees_.insert_or_assign(id,std::move(next));
        return true;
    }
    bool erase(::vs::spatial::region_id id,std::uint64_t revision) {
        if(!revision)return false;
        const auto it=trees_.find(id);
        if(it==trees_.end() || revision<=it->second.revision)return false;
        trees_.erase(it);
        return true;
    }
    const tree* find(::vs::spatial::region_id id) const {
        const auto it=trees_.find(id);
        return it==trees_.end()?nullptr:&it->second;
    }
    std::size_t size() const noexcept {return trees_.size();}
    void clear(){trees_.clear();}
    template<class Visit>void each(Visit&& visit) const {
        for(const auto& [id,tree]:trees_)visit(id,tree);
    }
private:
    std::unordered_map<::vs::spatial::region_id,tree,
                       ::vs::spatial::region_hash> trees_;
};
} // namespace vs::bvh
