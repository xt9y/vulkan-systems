#include <climits>
#include <limits>
#include <memory>
#include <stb_image.h>
#include <stdexcept>
#include <vs/assets/texture.hpp>

namespace vs::assets
{
std::span<const std::uint8_t> rgba(const scene::texture &image,
                                   std::vector<std::uint8_t> &scratch)
{
    scratch.clear();

    if (!image.width || !image.height ||
        image.width > INT_MAX || image.height > INT_MAX ||
        std::size_t(image.width) > std::numeric_limits<std::size_t>::max() /
                                       std::size_t(image.height) / 4u)
        throw std::invalid_argument("Invalid RGBA texture size");
    const std::size_t count = std::size_t(image.width) * image.height * 4u;

    if (image.pixels && !image.pixels->empty()) {
        if (image.pixels->size() != count)
            throw std::invalid_argument("Invalid decoded RGBA size");

        return *image.pixels;
    }

    if (!image.encoded || image.encoded->empty() || image.encoded->size() > INT_MAX)
        throw std::invalid_argument("Missing encoded texture bytes");

    int width = 0, height = 0, channels = 0;

    std::unique_ptr<stbi_uc, decltype(&stbi_image_free)> decoded(
        stbi_load_from_memory(image.encoded->data(), int(image.encoded->size()),
                              &width, &height, &channels, 4),
        stbi_image_free);
    if (!decoded || width != int(image.width) || height != int(image.height))
        throw std::invalid_argument("Texture dimensions differ from encoding");
    scratch.assign(decoded.get(), decoded.get() + count);

    return scratch;
}
} // namespace vs::assets
