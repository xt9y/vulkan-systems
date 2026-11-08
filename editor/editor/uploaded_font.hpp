#pragma once

// Exact white-pixel mask extracted from the user-supplied x_font.png.
// Original canvas: 144x256, 16x16 CP437 glyphs, 9x16 pixels per cell.
// Losslessly packed to a 1-bit PNG. Decoded once during editor startup.
// Original user PNG SHA256: c7265f68e7482d99d7279193f5aa8439b79196a9204ca1af5c30702f14fc4abd
// Packed PNG SHA256: 521a8ad7b53ee190a8a1be2e0b7994b56d30725ef8f31fdf179c5d3fdda99af1
namespace editor::vulkan::uploaded_font
{
    inline constexpr const char* encoded_png =
        "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";
}
