#pragma once
#include <vs/render/mesh.hpp>
#include <vs/gpu/resource.hpp>
#include <vs/gpu/upload.hpp>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace vs::render
{
    // Reusable GPU geometry ownership, independent of vertex layout and game assets.
    // Uploads are queued without blocking; ::vs::gpu::upload::retire() manages staging.
    class geometry
    {
    public:
        geometry()=default;
        geometry(const geometry&)=delete;
        geometry& operator=(const geometry&)=delete;
        geometry(geometry&&)=default;
        geometry& operator=(geometry&&)=default;

        void upload(::vs::gpu::upload::batch& transfers,
                    const void* vertices,std::size_t vertex_bytes,
                    const void* indices,std::size_t index_bytes,
                    std::uint32_t index_count,
                    vk::IndexType index_type=vk::IndexType::eUint32)
        {
            if(!vertices || !vertex_bytes || !indices || !index_bytes ||
               !index_count || (index_bytes % index_count)!=0)
                throw std::invalid_argument("Invalid indexed geometry upload");
            ::vs::gpu::owned_buffer next_vertices(transfers.add(
                vertices,vertex_bytes,vk::BufferUsageFlagBits::eVertexBuffer));
            ::vs::gpu::owned_buffer next_indices(transfers.add(
                indices,index_bytes,vk::BufferUsageFlagBits::eIndexBuffer));
            vertices_=std::move(next_vertices);
            borrowed_vertices_=nullptr;
            indices_=std::move(next_indices);
            index_count_=index_count;
            index_type_=index_type;
        }

        // Reuse an existing immutable vertex stream for simplified index LODs.
        // The source geometry must outlive the index-only variant.
        void upload_indices(::vs::gpu::upload::batch& transfers,
                            const geometry& source,
                            const void* indices,std::size_t index_bytes,
                            std::uint32_t index_count,
                            vk::IndexType index_type=vk::IndexType::eUint32)
        {
            if(!source.vertex_buffer() || !indices || !index_bytes ||
               !index_count || index_bytes%index_count)
                throw std::invalid_argument("Invalid LOD index upload");
            indices_=transfers.add(indices,index_bytes,
                vk::BufferUsageFlagBits::eIndexBuffer);
            vertices_.reset();
            borrowed_vertices_=source.vertex_buffer();
            index_count_=index_count;
            index_type_=index_type;
        }

        mesh view(std::uint32_t vertex_binding=0) const
        {
            mesh result{};
            if(vertex_buffer())
                result.streams.push_back({vertex_binding,vertex_buffer(),0});
            result.index_buffer=indices_.handle;
            result.index_type=index_type_;
            result.index_count=index_count_;
            return result;
        }

        void reset() noexcept
        {
            vertices_.reset();
            indices_.reset();
            borrowed_vertices_=nullptr;
            index_count_=0;
        }
        vk::Buffer vertex_buffer() const noexcept {
            return vertices_.handle ? vertices_.handle : borrowed_vertices_;
        }
        vk::Buffer index_buffer() const noexcept {return indices_.handle;}
        std::uint32_t index_count() const noexcept {return index_count_;}
    private:
        ::vs::gpu::owned_buffer vertices_,indices_;
        vk::Buffer borrowed_vertices_{};
        std::uint32_t index_count_=0;
        vk::IndexType index_type_=vk::IndexType::eUint32;
    };
}
