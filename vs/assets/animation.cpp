#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <ozz/animation/runtime/animation.h>
#include <ozz/animation/runtime/blending_job.h>
#include <ozz/animation/runtime/local_to_model_job.h>
#include <ozz/animation/runtime/sampling_job.h>
#include <ozz/animation/runtime/skeleton.h>
#include <ozz/base/io/archive.h>
#include <ozz/base/io/stream.h>
#include <ozz/base/maths/simd_math.h>
#include <ozz/base/maths/soa_transform.h>
#include <stdexcept>
#include <string_view>
#include <vector>
#include <vk/vk.hpp>
#include <vs/assets/animation.hpp>
#include <vs/scene/scene.hpp>

namespace vs::animation
{
namespace
{
struct runtime
{
    std::uint32_t id = 0;
    ozz::animation::Skeleton skeleton;
    std::vector<ozz::animation::Animation> clips;
    ozz::animation::SamplingJob::Context context;
    std::vector<ozz::math::SoaTransform> locals;
    std::vector<ozz::math::SoaTransform> secondary;
    std::vector<ozz::math::SoaTransform> blended;
    std::vector<ozz::math::SimdFloat4> mask;
    std::vector<ozz::math::Float4x4> models;
    std::vector<matrix> inverse_bind;
};

std::vector<std::unique_ptr<runtime>> &skins()
{
    static std::vector<std::unique_ptr<runtime>> storage;

    return storage;
}

template <class T>
void read(const std::vector<std::uint8_t> &bytes, T &output)
{
    if (bytes.empty()) throw std::runtime_error("Empty Ozz archive");
    ozz::io::MemoryStream stream;

    if (stream.Write(bytes.data(), bytes.size()) != bytes.size() || stream.Seek(0, ozz::io::Stream::kSet))
        throw std::runtime_error("Invalid Ozz archive stream");
    ozz::io::IArchive archive(&stream);

    if (!archive.TestTag<T>())
        throw std::runtime_error("Ozz archive tag mismatch");
    archive >> output;
}

runtime *find(std::uint32_t id)
{
    for (auto &item : skins())
        if (item->id == id) return item.get();

    return nullptr;
}

matrix multiply(const matrix &lhs, const matrix &rhs)
{
    matrix result{};

    for (int column = 0; column < 4; ++column)
        for (int row = 0; row < 4; ++row)
            for (int k = 0; k < 4; ++k)
                result[column * 4 + row] += lhs[k * 4 + row] * rhs[column * 4 + k];

    return result;
}
} // namespace
void synchronize(std::span<const ::vs::scene::skin> data)
{
    for (const auto &incoming : data)
    {
        if (find(incoming.id)) continue;

        auto item = std::make_unique<runtime>();
        item->id = incoming.id;
        read(incoming.skeleton, item->skeleton);

        if (item->skeleton.num_joints() == 0 || incoming.inverse_bind.size() != std::size_t(item->skeleton.num_joints()))
            throw std::runtime_error("Invalid streamed skin joint count");
        item->inverse_bind = incoming.inverse_bind;
        item->clips.resize(incoming.clips.size());

        for (std::size_t i = 0; i < incoming.clips.size(); ++i)
        {
            read(incoming.clips[i].bytes, item->clips[i]);

            if (item->clips[i].num_tracks() != item->skeleton.num_joints())
                throw std::runtime_error("Ozz clip does not match skin skeleton");
        }

        item->context.Resize(item->skeleton.num_joints());
        item->locals.resize(item->skeleton.num_soa_joints());
        item->secondary.resize(item->skeleton.num_soa_joints());
        item->blended.resize(item->skeleton.num_soa_joints());
        item->mask.resize(item->skeleton.num_soa_joints());
        item->models.resize(item->skeleton.num_joints());
        skins().push_back(std::move(item));
    }
}

std::uint32_t joints(std::uint32_t skin_id)
{
    const auto *item = find(skin_id);

    return item ? std::uint32_t(item->skeleton.num_joints()) : 0;
}

bool sample(std::uint32_t skin_id, std::uint32_t clip, float seconds, std::vector<matrix> &palette, std::uint32_t overlay, float weight, std::uint32_t mask_root, bool loop)
{
    auto *item = find(skin_id);
    if (!item || !std::isfinite(seconds) || !std::isfinite(weight)) return false;

    auto evaluate = [&](std::uint32_t index, std::vector<ozz::math::SoaTransform> &output)
    {
        if (item->clips.empty()) {
            std::copy(item->skeleton.joint_rest_poses().begin(), item->skeleton.joint_rest_poses().end(), output.begin());

            return true;
        }

        if (index >= item->clips.size()) return false;
        const auto &animation = item->clips[index];

        if (animation.duration() <= 0) return false;

        float t = loop ? std::fmod(seconds, animation.duration())
                       : std::clamp(seconds, 0.0f, animation.duration());

        if (t < 0) t += animation.duration();
        ozz::animation::SamplingJob job;
        job.animation = &animation;
        job.context = &item->context;
        job.ratio = t / animation.duration();
        job.output = ozz::make_span(output);

        return job.Run();
    };

    if (!evaluate(clip, item->locals)) return false;

    bool blending = overlay != UINT32_MAX && weight > 0.0f;

    if (blending) {
        if (!evaluate(overlay, item->secondary)) return false;
        weight = std::clamp(weight, 0.0f, 1.0f);

        const auto parents = item->skeleton.joint_parents();

        if (mask_root != UINT32_MAX) {
            if (mask_root >= std::uint32_t(item->skeleton.num_joints()))
                return false;

            for (std::size_t i = 0; i < item->mask.size(); ++i)
            {
                float lane[4]{};

                for (std::size_t j = 0; j < 4; ++j)
                {
                    auto bone = int(i * 4 + j);

                    while (bone >= 0 && bone < int(parents.size()))
                    {
                        if (std::uint32_t(bone) == mask_root) {
                            lane[j] = 1.0f;

                            break;
                        }

                        bone = parents[bone];
                    }
                }

                item->mask[i] = ozz::math::simd_float4::Load(lane[0], lane[1], lane[2], lane[3]);
            }
        }

        std::array<ozz::animation::BlendingJob::Layer, 2> layers{};
        layers[0].weight = std::max(0.0001f, 1.0f - weight);
        layers[0].transform = ozz::make_span(item->locals);
        layers[1].weight = weight;
        layers[1].transform = ozz::make_span(item->secondary);

        if (mask_root != UINT32_MAX) layers[1].joint_weights = ozz::make_span(item->mask);
        ozz::animation::BlendingJob job;
        job.threshold = 0.00001f;
        job.layers = ozz::make_span(layers);
        job.rest_pose = item->skeleton.joint_rest_poses();
        job.output = ozz::make_span(item->blended);

        if (!job.Run()) return false;
    }

    ozz::animation::LocalToModelJob job;
    job.skeleton = &item->skeleton;
    job.input = blending ? ozz::make_span(item->blended) : ozz::make_span(item->locals);
    job.output = ozz::make_span(item->models);

    if (!job.Run()) return false;
    palette.resize(item->models.size());

    for (std::size_t i = 0; i < item->models.size(); ++i)
    {
        matrix world{};

        for (int c = 0; c < 4; ++c)
            ozz::math::StorePtrU(item->models[i].cols[c], world.data() + c * 4);
        palette[i] = multiply(world, item->inverse_bind[i]);
    }

    return true;
}

float duration(std::uint32_t skin_id, std::uint32_t clip)
{
    const auto *skin = find(skin_id);

    return skin && clip < skin->clips.size() ? skin->clips[clip].duration() : 0.f;
}

std::uint32_t joint(std::uint32_t skin_id, std::string_view name)
{
    const auto *skin = find(skin_id);
    if (!skin) return UINT32_MAX;

    const auto names = skin->skeleton.joint_names();

    for (std::uint32_t i = 0; i < names.size(); ++i)
        if (name == names[i]) return i;

    return UINT32_MAX;
}

bool joint_transform(std::uint32_t skin_id, std::uint32_t index, matrix &out)
{
    const auto *skin = find(skin_id);
    if (!skin || index >= skin->models.size()) return false;

    for (int c = 0; c < 4; ++c)
        ozz::math::StorePtrU(skin->models[index].cols[c], out.data() + c * 4);

    return true;
}

bool offset_joint(std::uint32_t skin_id, std::uint32_t root, vk::util::math::vec3 offset, std::vector<matrix> &palette)
{
    auto *skin = find(skin_id);
    if (!skin || root >= skin->models.size() || palette.size() != skin->models.size())
        return false;

    const auto parents = skin->skeleton.joint_parents();

    for (std::size_t i = 0; i < skin->models.size(); ++i)
    {
        int ancestor = int(i);

        while (ancestor >= 0 && std::uint32_t(ancestor) != root)
            ancestor = parents[ancestor];

        if (ancestor < 0)
            continue;
        auto &model = skin->models[i];
        model.cols[3] = model.cols[3] + ozz::math::simd_float4::Load(offset.x, offset.y, offset.z, 0);
        matrix world{};

        for (int c = 0; c < 4; ++c)
            ozz::math::StorePtrU(model.cols[c], world.data() + c * 4);
        palette[i] = multiply(world, skin->inverse_bind[i]);
    }

    return true;
}

void reset()
{
    skins().clear();
}

} // namespace vs::animation
