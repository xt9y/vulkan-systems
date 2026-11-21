#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>
#include <vs/editor/editor.hpp>

namespace vs::editor
{
// Immediate-mode drawing in the editor's 960x540 virtual canvas.
// Independent of the backend: Vulkan converts these into draw_commands.
class canvas
{
public:
    struct slice
    {
        double value = 0;

        color tint{};
    };

    explicit canvas(draw_list &output) : out_(output) {}
    void rect(int x, int y, int w, int h, color tint)
    {
        if (w <= 0 || h <= 0) return;
        draw_command d{};
        d.kind = draw_command::type::rectangle;
        d.x = x;
        d.y = y;
        d.width = w;
        d.height = h;
        d.tint = tint;
        out_.push_back(std::move(d));
    }

    void label(int x, int y, std::string text, color tint = {.85f, .87f, .9f, 1.f},
               draw_command::alignment alignment = draw_command::alignment::left)
    {
        draw_command d{};
        d.kind = draw_command::type::text;
        d.x = x;
        d.y = y;
        d.text = std::move(text);
        d.tint = tint;
        d.text_alignment = alignment;
        out_.push_back(std::move(d));
    }

    void panel(int x, int y, int w, int h, color background = {.07f, .07f, .095f, 1.f},
               color border = {.22f, .27f, .34f, 1.f})
    {
        rect(x, y, w, h, background);
        rect(x, y, w, 2, border);
        rect(x, y + h - 2, w, 2, border);
        rect(x, y, 2, h, border);
        rect(x + w - 2, y, 2, h, border);
    }

    void bar(int x, int y, int w, int h, double fraction, color filled,
             color empty = {.15f, .17f, .21f, 1.f})
    {
        rect(x, y, w, h, empty);
        rect(x, y, std::clamp(int(std::lround(w * std::clamp(fraction, 0., 1.))), 0, w), h, filled);
    }

    void segment(int x0, int y0, int x1, int y1, color tint, int thickness = 2)
    {
        const int steps = std::max(std::abs(x1 - x0), std::abs(y1 - y0));

        if (!steps) {
            rect(x0, y0, thickness, thickness, tint);

            return;
        }

        for (int i = 0; i <= steps; ++i)
        {
            const double t = double(i) / steps;
            rect(int(std::lround(x0 + (x1 - x0) * t)),
                 int(std::lround(y0 + (y1 - y0) * t)), thickness, thickness, tint);
        }
    }

    void graph(int x, int y, int w, int h, const std::vector<double> &values,
               double upper, color ink = {.35f, .8f, .65f, 1.f})
    {
        rect(x, y, w, h, {.10f, .12f, .17f, 1.f});

        for (int i = 1; i < 4; ++i)
            rect(x, y + h * i / 4, w, 1, {.23f, .27f, .32f, 1.f});

        if (values.size() < 2 || w < 2 || h < 2 || upper <= 0.) return;
        const std::size_t sample_count = std::min(values.size(), std::size_t(w));
        const std::size_t first = values.size() - sample_count;

        auto coord = [&](std::size_t i)
        {
            const int sx = x + int(i * std::size_t(w - 2) / (sample_count - 1));
            const double v = std::clamp(values[first + i] / upper, 0., 1.);

            return std::pair{sx, y + h - 2 - int(v * double(h - 3))};
        };

        auto prev = coord(0);

        for (std::size_t i = 1; i < sample_count; ++i)
        {
            const auto curr = coord(i);
            segment(prev.first, prev.second, curr.first, curr.second, ink);
            prev = curr;
        }
    }

    void pie(int x, int y, int radius, const std::vector<slice> &parts)
    {
        if (radius <= 0 || parts.empty()) return;

        double total = 0;

        for (const auto &p : parts)
            total += std::max(0., p.value);

        if (total <= 0) return;
        std::vector<double> ends;
        ends.reserve(parts.size());

        double sum = 0;

        for (const auto &p : parts)
        {
            sum += std::max(0., p.value) / total;
            ends.push_back(sum);
        }

        constexpr double tau = 6.283185307179586;
        // Coalesce equal-color scanline runs into ordinary editor rectangles.
        for (int row = -radius; row <= radius; ++row)
        {
            const int half = int(std::sqrt(double(radius * radius - row * row)));
            int from = -half;

            while (from <= half)
            {
                const auto sector = [&](int cx)
                {
                    double angle = std::atan2(double(row), double(cx)) / tau + .25;

                    if (angle < 0) angle += 1.;

                    auto it = std::lower_bound(ends.begin(), ends.end(), angle);

                    return std::min<std::size_t>(it - ends.begin(), parts.size() - 1);
                };

                const std::size_t id = sector(from);

                int last = from;

                while (last < half && sector(last + 1) == id)
                    ++last;
                rect(x + from, y + row, last - from + 1, 1, parts[id].tint);
                from = last + 1;
            }
        }
    }

private:
    draw_list &out_;
};
} // namespace vs::editor
