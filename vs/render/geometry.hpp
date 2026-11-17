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
            indices_=std::move(next_indices);
            index_count_=index_count;
            index_type_=index_type;
        }

        mesh view(std::uint32_t vertex_binding=0) const
        {
            mesh result{};
            if(vertices_.handle)
                result.streams.push_back({vertex_binding,vertices_.handle,0});
            result.index_buffer=indices_.handle;
            result.index_type=index_type_;
            result.index_count=index_count_;
            return result;
        }

        void reset() noexcept
        {
            vertices_.reset();
            indices_.reset();
            index_count_=0;
        }
        vk::Buffer vertex_buffer() const noexcept {return vertices_.handle;}
        vk::Buffer index_buffer() const noexcept {return indices_.handle;}
        std::uint32_t index_count() const noexcept {return index_count_;}
    private:
        ::vs::gpu::owned_buffer vertices_,indices_;
        std::uint32_t index_count_=0;
        vk::IndexType index_type_=vk::IndexType::eUint32;
    };
}
