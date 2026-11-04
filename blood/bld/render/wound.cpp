#include "wound.hpp"
#include "../core/system.hpp"
#include "../surfaces/wounds.hpp"
#include <stdexcept>
#include <algorithm>
#include <cstring>
#include <limits>

namespace bld::render::wound
{
    namespace { state *data = nullptr; }

    state *get() { return data; }

    void init(std::uint32_t frames_in_flight)
    {
        if (data) throw std::logic_error("Blood wound GPU already initialized");
        if (!frames_in_flight) throw std::invalid_argument("Zero frame count");
        data = new state{};
        data->frames = frames_in_flight;
        try
        {
            const auto flags = vk::MemoryPropertyFlagBits::eHostVisible |
                               vk::MemoryPropertyFlagBits::eHostCoherent;
            const auto system = bld::system::get();
            const auto width = system ? system->options.wound_resolution : 128u;
            std::vector<std::uint32_t> zeros(
                std::size_t(width) * width / 4, 0u);
            data->fallback = vk::buffer::create(
                zeros.size() * sizeof(std::uint32_t),
                vk::BufferUsageFlagBits::eStorageBuffer, flags);
            vk::buffer::write(data->fallback, zeros.data(),
                              zeros.size() * sizeof(std::uint32_t));
        }
        catch (...)
        {
            shutdown();
            throw;
        }
    }

    vk::DescriptorBufferInfo descriptor(instance_id id, std::uint32_t frame_index)
    {
        if (!data || frame_index >= data->frames)
            throw std::logic_error("Blood wound GPU not initialized or bad frame");
        const auto *atlas = bld::wounds::find(id);
        if (!atlas || atlas->pixels.empty())
            return vk::buffer::descriptor(data->fallback);

        auto &item = data->instances[id];
        if (item.buffers.empty())
        {
            item.buffers.resize(data->frames);
            item.generation.assign(data->frames, ~std::uint64_t{0});
        }
        auto &buffer = item.buffers.at(frame_index);
        const auto bytes = atlas->pixels.size() * sizeof(atlas->pixels.front());
        if (!buffer.handle || buffer.size < bytes)
        {
            if (buffer.handle) vk::buffer::destroy(buffer);
            buffer = vk::buffer::create(
                bytes, vk::BufferUsageFlagBits::eStorageBuffer,
                vk::MemoryPropertyFlagBits::eHostVisible |
                vk::MemoryPropertyFlagBits::eHostCoherent);
            item.generation[frame_index] = ~std::uint64_t{0};
        }
        if (item.generation[frame_index] != atlas->generation)
        {
            vk::buffer::write(buffer, atlas->pixels.data(), bytes);
            item.generation[frame_index] = atlas->generation;
        }
        return vk::buffer::descriptor(buffer);
    }

    bool has_wounds(const std::vector<instance_id> &instances)
    {
        for (auto id : instances)
        {
            const auto *atlas = bld::wounds::find(id);
            if (atlas && !atlas->pixels.empty()) return true;
        }
        return false;
    }

    vk::DescriptorBufferInfo descriptor_batch(
        std::uint32_t mesh_id, const std::vector<instance_id> &instances,
        std::uint32_t frame_index)
    {
        if (!data || frame_index >= data->frames || !mesh_id)
            throw std::invalid_argument("Invalid blood wound batch");
        if (!has_wounds(instances)) return vk::buffer::descriptor(data->fallback);

        const auto *system = bld::system::get();
        const auto width = system ? system->options.wound_resolution : 128u;
        const auto words = std::size_t(width) * width / 4;
        const auto bytes = std::max<std::size_t>(
            sizeof(std::uint32_t), instances.size() * words * sizeof(std::uint32_t));

        auto &batch = data->batches[mesh_id];
        if (batch.buffers.empty())
        {
            batch.buffers.resize(data->frames);
            batch.generation.assign(data->frames, ~std::uint64_t{0});
        }
        auto &buffer = batch.buffers[frame_index];
        if (!buffer.handle || buffer.size < bytes)
        {
            if (buffer.handle) vk::buffer::destroy(buffer);
            buffer = vk::buffer::create(
                bytes, vk::BufferUsageFlagBits::eStorageBuffer,
                vk::MemoryPropertyFlagBits::eHostVisible |
                vk::MemoryPropertyFlagBits::eHostCoherent);
            batch.generation[frame_index] = ~std::uint64_t{0};
        }

        std::uint64_t signature = 14695981039346656037ull;
        for (auto id : instances)
        {
            const auto *a = bld::wounds::find(id);
            signature ^= std::uint64_t(id);
            signature *= 1099511628211ull;
            signature ^= a ? a->generation : 0;
            signature *= 1099511628211ull;
        }
        if (batch.generation[frame_index] != signature)
        {
            std::vector<std::uint32_t> pixels(instances.size() * words, 0);
            for (std::size_t i = 0; i < instances.size(); ++i)
            {
                const auto *a = bld::wounds::find(instances[i]);
                if (a && a->pixels.size() == words)
                    std::copy(a->pixels.begin(), a->pixels.end(),
                              pixels.begin() + i * words);
            }
            vk::buffer::write(buffer, pixels.data(), pixels.size() * sizeof(std::uint32_t));
            batch.generation[frame_index] = signature;
        }
        return vk::buffer::descriptor(buffer);
    }

    void clear_batches()
    {
        if (!data) return;
        for (auto &entry : data->batches)
            for (auto &buffer : entry.second.buffers)
                if (buffer.handle) vk::buffer::destroy(buffer);
        data->batches.clear();
    }

    void remove(instance_id id)
    {
        if (!data) return;
        const auto it = data->instances.find(id);
        if (it == data->instances.end()) return;
        for (auto &buffer : it->second.buffers)
            if (buffer.handle) vk::buffer::destroy(buffer);
        data->instances.erase(it);
    }

    void shutdown()
    {
        if (!data) return;
        for (auto &entry : data->instances)
            for (auto &buffer : entry.second.buffers)
                if (buffer.handle) vk::buffer::destroy(buffer);
        clear_batches();
        if (data->fallback.handle) vk::buffer::destroy(data->fallback);
        delete data;
        data = nullptr;
    }
}
