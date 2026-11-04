#include "wounds.hpp"
#include "decals.hpp"
#include "../math/geometry.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace bld::wounds
{
    namespace { state *data = nullptr; }

    state *get() { return data; }

    void init(const settings &options)
    {
        if (data) throw std::logic_error("Blood wounds already initialized");
        data = new state{};
        data->resolution = std::max(16u, options.wound_resolution);
        data->max_marks = options.max_wounds;
        data->max_instances = options.max_instances;
    }

    const atlas *find(instance_id id)
    {
        if (!data) return nullptr;
        const auto it = data->atlases.find(id);
        return it == data->atlases.end() ? nullptr : &it->second;
    }

    void stamp(instance_id target, vec2 uv, float radius, float strength,
               std::uint32_t seed)
    {
        if (!data || !target || !data->max_marks ||
            uv.x < 0 || uv.y < 0 || uv.x > 1 || uv.y > 1) return;
        if (data->marks >= data->max_marks) return;
        if (data->atlases.size() >= data->max_instances &&
            data->atlases.find(target) == data->atlases.end()) return;

        auto &atlas = data->atlases[target];
        atlas.target = target;
        atlas.resolution = data->resolution;
        const auto side = int(atlas.resolution);
        const auto words = std::size_t(side) * side / 4;
        if (atlas.pixels.size() != words) atlas.pixels.assign(words, 0);
        const float cx = uv.x * float(side - 1);
        const float cy = uv.y * float(side - 1);
        const float size = std::max(1.6f, radius * float(side));
        const int reach = int(std::ceil(size * 1.3f));
        const float angle = math::unit(seed) * 6.2831853f;
        for (int y = std::max(0, int(cy) - reach);
             y <= std::min(side - 1, int(cy) + reach); ++y)
        {
            for (int x = std::max(0, int(cx) - reach);
                 x <= std::min(side - 1, int(cx) + reach); ++x)
            {
                const float dx = (float(x) + .5f - cx) / size;
                const float dy = (float(y) + .5f - cy) / size;
                const float a = std::atan2(dy, dx) - angle;
                const float edge = .79f + .11f*std::sin(a*7.f) +
                                   .06f*std::sin(a*13.f);
                const float r = std::sqrt(dx*dx + dy*dy);
                const float mask = std::max(0.f, std::min(1.f,
                    (edge-r)*4.f))*std::max(0.f, strength);
                const auto cover = std::uint32_t(std::min(15.f, mask*15.f));
                const std::size_t pixel = std::size_t(y)*side + std::size_t(x);
                const std::size_t word = pixel / 4;
                const auto shift = std::uint32_t((pixel%4)*8);
                const auto prior = (atlas.pixels[word] >> shift)&0xffu;
                const auto next = std::max(cover, prior >> 4u);
                const auto wet = std::max(cover, prior & 15u);
                atlas.pixels[word] = (atlas.pixels[word] & ~(255u << shift)) |
                                     ((next*16u + wet) << shift);
            }
        }
        ++atlas.generation;
        ++data->marks;
    }

    void source_at(instance_id target, vec3 local_position, float strength,
                   std::uint32_t seed)
    {
        if (!data || !target || data->sources.size() >= data->max_marks) return;
        data->sources.push_back({target, local_position, 0, .7f,
                                 strength, seed});
    }

    std::vector<vec3> update(float dt)
    {
        std::vector<vec3> emission;
        if (!data) return emission;
        for (auto &s : data->sources)
        {
            s.age += dt;
            if (s.age > 9.f || s.age < s.next_drip) continue;
            const auto transforms = surfaces::get();
            auto it = transforms ? transforms->transforms.find(s.target) :
                                   std::unordered_map<instance_id, mat4>::const_iterator{};
            if (!transforms || it == transforms->transforms.end()) continue;
            emission.push_back(math::transform_point(it->second, s.local_position));
            s.next_drip += .6f + math::unit(s.seed) * 1.4f;
        }
        data->sources.erase(std::remove_if(data->sources.begin(), data->sources.end(),
            [](const source &s) { return s.age > 9.f; }), data->sources.end());
        // Wound shape persists: wetness is allowed to dry gradually.
        for (auto &pair : data->atlases)
        {
            auto &a = pair.second;
            a.elapsed += dt;
            if (a.elapsed < 1.f) continue;
            a.elapsed = 0;
            for (auto &word : a.pixels)
            {
                for (std::uint32_t bit = 0; bit < 32; bit += 8)
                {
                    const auto wet = (word >> bit) & 15u;
                    if (wet) word -= (1u << bit);
                }
            }
            ++a.generation;
        }
        return emission;
    }

    void remove(instance_id id)
    {
        if (!data) return;
        data->atlases.erase(id);
        auto &s = data->sources;
        s.erase(std::remove_if(s.begin(), s.end(),
            [id](const source &a) { return a.target == id; }), s.end());
    }

    void shutdown()
    {
        delete data;
        data = nullptr;
    }
}
