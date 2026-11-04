#include "events.hpp"
#include "../core/system.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace bld::network
{
    namespace
    {
        state *data = nullptr;

        void u32(std::uint8_t *p, std::uint32_t value)
        {
            for (int i = 0; i < 4; ++i) p[i] = std::uint8_t(value >> (i*8));
        }

        void u64(std::uint8_t *p, std::uint64_t value)
        {
            for (int i = 0; i < 8; ++i) p[i] = std::uint8_t(value >> (i*8));
        }

        std::uint32_t read32(const std::uint8_t *p)
        {
            std::uint32_t n = 0;
            for (int i = 0; i < 4; ++i) n |= std::uint32_t(p[i]) << (i*8);
            return n;
        }

        std::uint64_t read64(const std::uint8_t *p)
        {
            std::uint64_t n = 0;
            for (int i = 0; i < 8; ++i) n |= std::uint64_t(p[i]) << (i*8);
            return n;
        }

        void f32(std::uint8_t *p, float value)
        {
            std::uint32_t bits;
            std::memcpy(&bits, &value, 4);
            u32(p, bits);
        }

        float read_float(const std::uint8_t *p)
        {
            const auto bits = read32(p);
            float value;
            std::memcpy(&value, &bits, 4);
            return value;
        }

        std::uint32_t checksum(const std::uint8_t *p, std::size_t size)
        {
            std::uint32_t value = 2166136261u;
            for (std::size_t i = 0; i < size; ++i)
                value = (value ^ p[i]) * 16777619u;
            return value;
        }

        bool finite(vec3 p)
        {
            return std::isfinite(p.x) &&
                   std::isfinite(p.y) && std::isfinite(p.z);
        }
    }

    state *get() { return data; }

    void init(std::size_t replay_window)
    {
        if (data) throw std::logic_error("Blood network already initialized");
        data = new state{};
        data->replay_window = std::max<std::size_t>(1, replay_window);
    }

    packet encode(const impact &event)
    {
        packet out{};
        out[0] = 'B'; out[1] = 'L'; out[2] = 'D'; out[3] = '1';
        out[4] = 1;
        out[5] = event.uv_valid ? 1 : 0;
        out[6] = std::uint8_t(event.type);
        u64(out.data() + 8, event.id);
        u32(out.data() + 16, event.source);
        u32(out.data() + 20, event.target);
        u32(out.data() + 24, event.seed);
        u32(out.data() + 28, event.tick);
        f32(out.data() + 32, event.position.x);
        f32(out.data() + 36, event.position.y);
        f32(out.data() + 40, event.position.z);
        f32(out.data() + 44, event.normal.x);
        f32(out.data() + 48, event.normal.y);
        f32(out.data() + 52, event.normal.z);
        f32(out.data() + 56, event.direction.x);
        f32(out.data() + 60, event.direction.y);
        f32(out.data() + 64, event.direction.z);
        f32(out.data() + 68, event.uv.x);
        f32(out.data() + 72, event.uv.y);
        f32(out.data() + 76, event.energy);
        u32(out.data() + 80, checksum(out.data(), 80));
        return out;
    }

    std::optional<impact> decode(const void *bytes, std::size_t count)
    {
        if (!bytes || count != packet{}.size()) return std::nullopt;
        const auto *p = static_cast<const std::uint8_t *>(bytes);
        if (p[0] != 'B' || p[1] != 'L' || p[2] != 'D' || p[3] != '1' ||
            p[4] != 1 || (p[5] & ~1u) || p[7] ||
            p[6] > std::uint8_t(material::glass) ||
            checksum(p, 80) != read32(p + 80)) return std::nullopt;
        impact event{};
        event.uv_valid = p[5] != 0;
        event.type = material(p[6]);
        event.id = read64(p + 8);
        event.source = read32(p + 16);
        event.target = read32(p + 20);
        event.seed = read32(p + 24);
        event.tick = read32(p + 28);
        event.position = {
            read_float(p + 32), read_float(p + 36), read_float(p + 40)
        };
        event.normal = {
            read_float(p + 44), read_float(p + 48), read_float(p + 52)
        };
        event.direction = {
            read_float(p + 56), read_float(p + 60), read_float(p + 64)
        };
        event.uv = {read_float(p + 68), read_float(p + 72)};
        event.energy = read_float(p + 76);
        if (!event.id || !finite(event.position) ||
            !finite(event.normal) || !finite(event.direction) ||
            !std::isfinite(event.energy) || event.energy <= 0.f ||
            event.energy > 10.f || !std::isfinite(event.uv.x) ||
            !std::isfinite(event.uv.y) ||
            (event.uv_valid && (event.uv.x < 0.f || event.uv.x > 1.f ||
                                event.uv.y < 0.f || event.uv.y > 1.f)))
            return std::nullopt;
        return event;
    }

    bool receive(const void *bytes, std::size_t count)
    {
        const auto value = decode(bytes, count);
        if (!value || !system::get()) return false;
        if (!data) init();
        std::lock_guard<std::mutex> guard(data->mutex);
        // The source ID is part of the replay key to accommodate multiple peers.
        const std::uint64_t key = value->id ^
            (std::uint64_t(value->source) * 0x9e3779b97f4a7c15ull);
        if (!data->seen.insert(key).second) return false;
        if (data->history.size() >= data->replay_window)
        {
            data->seen.erase(data->history.front());
            data->history.pop_front();
        }
        data->history.push_back(key);
        if (data->pending.size() >= data->replay_window)
            data->pending.pop_front();
        data->pending.push_back(*value);
        return true;
    }

    void dispatch()
    {
        if (!data) return;
        std::deque<impact> ready;
        {
            std::lock_guard<std::mutex> guard(data->mutex);
            ready.swap(data->pending);
        }
        for (const auto &event : ready)
            system::apply(event);
    }

    void reset()
    {
        if (!data) return;
        std::lock_guard<std::mutex> guard(data->mutex);
        data->history.clear();
        data->seen.clear();
        data->pending.clear();
    }

    void shutdown()
    {
        delete data;
        data = nullptr;
    }
}
