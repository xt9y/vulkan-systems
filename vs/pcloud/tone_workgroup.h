#ifndef VS_PCLOUD_TONE_WORKGROUP_H
#define VS_PCLOUD_TONE_WORKGROUP_H

#ifndef VS_PCLOUD_TONE_WORKGROUP_SIZE
#define VS_PCLOUD_TONE_WORKGROUP_SIZE 16
#endif

#ifdef __cplusplus
#include <cstdint>

namespace vs::pcloud
{
inline constexpr std::uint32_t tone_workgroup_size =
    VS_PCLOUD_TONE_WORKGROUP_SIZE;

constexpr std::uint32_t tone_workgroup_count(std::uint32_t pixels) noexcept
{
    return (pixels + tone_workgroup_size - 1u) / tone_workgroup_size;
}
} // namespace vs::pcloud
#endif

#endif
