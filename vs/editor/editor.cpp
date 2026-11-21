#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vs/editor/editor.hpp>

namespace vs::editor
{
namespace
{
// Original Hell2024: PRESENT_WIDTH=960, PRESENT_HEIGHT=540.
// Menu uses g_menuLocation=(76,460), margins=(16,20) and 200px width.
// Game host controls all object types/labels/properties.
constexpr int virtual_width = 960, virtual_height = 540;
constexpr int menu_width = 184, menu_height = 222;
constexpr int margin_x = 16, margin_y = 20;
constexpr int line_height = 16;
constexpr color white{0.80f, 0.80f, 0.83f, 1};
constexpr color teal{0.26f, 0.46f, 0.44f, 1};
constexpr color red{0.55f, 0.12f, 0.15f, 1};

struct rect
{ int x = 0, y = 0, w = 0, h = 0; };
bool inside(int x, int y, rect r)
{
    return x >= r.x && y >= r.y && x < r.x + r.w && y < r.y + r.h;
}

int virtual_x(int x, int viewport)
{
    return std::clamp(int(double(x) * virtual_width /
                          double(std::max(1, viewport))),
                      0, virtual_width);
}

int virtual_y(int y, int viewport)
{
    return std::clamp(int(double(y) * virtual_height /
                          double(std::max(1, viewport))),
                      0, virtual_height);
}

void rectangle(draw_list &out, rect area, color tint)
{
    if (area.w <= 0 || area.h <= 0) return;
    draw_command item{};
    item.kind = draw_command::type::rectangle;
    item.x = area.x;
    item.y = area.y;
    item.width = area.w;
    item.height = area.h;
    item.tint = tint;
    out.push_back(std::move(item));
}

void image(draw_list &out, draw_command::texture texture, rect area)
{
    if (area.w <= 0 || area.h <= 0) return;
    draw_command item{};
    item.kind = draw_command::type::image;
    item.image = texture;
    item.x = area.x;
    item.y = area.y;
    item.width = area.w;
    item.height = area.h;
    out.push_back(std::move(item));
}

void text(draw_list &out, int x, int y, std::string content,
          draw_command::alignment align = draw_command::alignment::left)
{
    draw_command item{};
    item.kind = draw_command::type::text;
    item.x = x;
    item.y = y;
    item.scale = 1;
    item.text_alignment = align;
    item.text = std::move(content);
    item.tint = white;
    out.push_back(std::move(item));
}

rect menu_rect(int x, int y)
{
    return {x, y, menu_width, menu_height};
}

rect label_rect(const screen_label &label, int width, int height)
{
    const int cx = virtual_x(label.x, width);
    const int cy = virtual_y(label.y, height);
    const int label_width = std::max(62,
                                     int(label.text.size()) * 8 + 12);
    constexpr int label_height = 22;
    // All textured world signs share the same centered anchor.
    const int top = cy - label_height / 2;

    return {std::clamp(cx - label_width / 2, 0, std::max(0, virtual_width - label_width)), std::clamp(top, 0, virtual_height - label_height), label_width, label_height};
}

std::string value_string(const property &p)
{
    if (p.kind == property_kind::boolean)
        return p.value != 0 ? "TRUE" : "FALSE";

    if (p.kind == property_kind::choice) {
        const auto index = int(std::round(p.value));

        return index >= 0 && index < int(p.choices.size())
                   ? p.choices[std::size_t(index)]
                   : "NONE";
    }

    std::ostringstream out;

    if (p.kind == property_kind::integer)
        out << std::llround(p.value);
    else
        out << std::fixed << std::setprecision(std::clamp(p.precision, 0, 6)) << p.value;

    return out.str();
}
} // namespace

void session::set_labels(std::vector<screen_label> labels)
{
    std::lock_guard<std::mutex> guard(mutex_);
    labels_ = std::move(labels);
}

void session::set_properties(std::uint64_t object, std::vector<property> values)
{
    std::lock_guard<std::mutex> guard(mutex_);

    if (auto edits = local_edits_.find(object); edits != local_edits_.end()) {
        const auto now = std::chrono::steady_clock::now();

        for (auto &p : values)
        {
            const auto it = edits->second.find(p.id);

            if (it == edits->second.end()) continue;

            const double tolerance = std::max(1e-4, p.step * .05);

            if (std::abs(p.value - it->second.value) <= tolerance ||
                now - it->second.updated > std::chrono::seconds(3))
                edits->second.erase(it);
            else
                p.value = it->second.value;
        }

        if (edits->second.empty()) local_edits_.erase(edits);
    }

    properties_[object] = std::move(values);

    if (menu_ == menu::properties && selected_object_ == object)
        selected_row_ = std::clamp(selected_row_, 0,
                                   int(properties_[object].size()));
}

void session::bind(std::uint64_t object, property description,
                   std::function<void(double)> apply)
{
    if (!object || description.id.empty() || !apply)
        throw std::invalid_argument("Editor binding needs an object, property and callback");
    std::lock_guard<std::mutex> guard(mutex_);
    auto &properties = properties_[object];

    auto found = std::find_if(properties.begin(), properties.end(),
                              [&](const property &p)
                              { return p.id == description.id; });
    const auto id = description.id;

    if (found != properties.end()) *found = std::move(description);
    else
        properties.push_back(std::move(description));
    bound_properties_[object][id] = std::move(apply);
}

void session::reset()
{
    std::lock_guard<std::mutex> guard(mutex_);
    visible_ = cursor_free_ = dragging_menu_ = delete_armed_ = pointer_down_ = false;
    visualization_ = 0;
    menu_ = menu::none;
    selected_row_ = scroll_first_ = 0;
    selected_object_ = 0;
    labels_.clear();
    properties_.clear();
    bound_properties_.clear();
    local_edits_.clear();
    pending_.clear();
}

void session::set_font_metrics(std::array<int, 128> widths, int line_height)
{
    std::lock_guard<std::mutex> guard(mutex_);
    font_widths_ = std::move(widths);
    line_height_ = std::max(1, line_height);
}

void session::set_frame_time(double milliseconds)
{
    if (!std::isfinite(milliseconds) || milliseconds <= 0) return;
    std::lock_guard<std::mutex> guard(mutex_);
    frame_time_ms_ = frame_time_ms_ > 0 ? frame_time_ms_ * 0.85 + milliseconds * 0.15 : milliseconds;
}

void session::press(key command, bool shift, bool accelerated)
{
    std::lock_guard<std::mutex> guard(mutex_);

    if (command == key::cursor && visible_) cursor_free_ = !cursor_free_;
    if (command == key::toggle) {
        visible_ = !visible_;

        if (!visible_) {
            cursor_free_ = false;
            visualization_ = 0;
        }

        menu_ = menu::none;
        selected_object_ = 0;
        dragging_menu_ = false;
        delete_armed_ = false;
        selected_row_ = scroll_first_ = 0;
    }

    if (!visible_) return;
    if (command >= key::view_1 && command <= key::view_9) {
        const auto next = std::uint32_t(command) - std::uint32_t(key::view_1) + 1u;
        visualization_ = visualization_ == next ? 0u : next;
        menu_ = menu::none;
        selected_object_ = 0;
        delete_armed_ = dragging_menu_ = false;
        // Always return camera control to the viewport for 3D inspection.
        cursor_free_ = false;

        return;
    }

    if (visualization_) return;
    if (command == key::cancel) {
        if (menu_ == menu::properties && delete_armed_ && selected_object_) {
            pending_.push_back({action_kind::remove, "", selected_object_, "", 0});
            local_edits_.erase(selected_object_);
            properties_.erase(selected_object_);
            bound_properties_.erase(selected_object_);
        }

        menu_ = menu::none;
        selected_object_ = 0;
        dragging_menu_ = false;
        delete_armed_ = false;

        return;
    }

    if (command == key::erase && selected_object_ && menu_ == menu::properties) {
        pending_.push_back({action_kind::remove, "", selected_object_, "", 0});
        local_edits_.erase(selected_object_);
        properties_.erase(selected_object_);
        bound_properties_.erase(selected_object_);
        selected_object_ = 0;
        menu_ = menu::none;

        return;
    }

    const int jump = shift ? 10 : 1;

    if (menu_ == menu::insert) {
        if (command == key::up) selected_row_ = std::max(0, selected_row_ - jump);
        if (command == key::down) selected_row_ = std::min(
                                      std::max(0, int(catalog_.size()) - 1), selected_row_ + jump);
        if (command == key::confirm && !catalog_.empty()) {
            pending_.push_back({action_kind::insert, catalog_[std::size_t(selected_row_)].id, 0, "", 0});
            menu_ = menu::none;
            scroll_first_ = 0;
        }
    }
    else if (menu_ == menu::properties) {
        const auto it = properties_.find(selected_object_);
        const int property_count = it != properties_.end() ? int(it->second.size()) : 0;

        if (command == key::up) selected_row_ = std::max(0, selected_row_ - jump);
        if (command == key::down) selected_row_ = std::min(
                                      property_count, selected_row_ + jump);
        if (command == key::left || command == key::right) {
            if (selected_row_ == property_count)
                delete_armed_ = command == key::right;
            else if (it != properties_.end()) {
                property &p = it->second[std::size_t(selected_row_)];

                const double before = p.value;

                if (p.kind == property_kind::boolean)
                    p.value = command == key::right ? 1.0 : 0.0;
                else {
                    const double upper = p.kind == property_kind::choice ? double(std::max(0, int(p.choices.size()) - 1)) : p.maximum;
                    const double step = p.step > 0 ? p.step : 1.0;
                    const double delta = step * (accelerated ? 100.0 : shift ? 10.0
                                                                             : 1.0);
                    p.value = std::clamp(before +
                                             (command == key::right ? delta : -delta),
                                         std::min(p.minimum, upper),
                                         std::max(p.minimum, upper));
                    if (p.kind == property_kind::integer ||
                        p.kind == property_kind::choice)
                        p.value = std::round(p.value);
                }

                if (p.value != before) {
                    local_edits_[selected_object_][p.id] = {
                        p.value, std::chrono::steady_clock::now()
                    };

                    pending_.push_back({action_kind::edit, "", selected_object_, p.id, p.value});
                }
            }
        }
    }
    else if (command == key::confirm) {
        menu_ = menu::insert;
        selected_row_ = scroll_first_ = 0;
    }
}

void session::pointer(int x, int y, int width, int height, bool down)
{
    std::lock_guard<std::mutex> guard(mutex_);

    const bool click = down && !pointer_down_;

    pointer_down_ = down;

    const bool inside_view = width > 0 && height > 0 &&
                             x >= 0 && y >= 0 && x < width && y < height;
    mouse_x_ = inside_view ? virtual_x(x, width) : -1;
    mouse_y_ = inside_view ? virtual_y(y, height) : -1;

    if (!down) dragging_menu_ = false;
    if (!visible_ || visualization_ || !inside_view) return;

    const int mx = virtual_x(x, width);
    const int my = virtual_y(y, height);

    if (dragging_menu_ && down) {
        menu_x_ = std::clamp(mx - drag_x_, 0, virtual_width - menu_width);
        menu_y_ = std::clamp(my - drag_y_, 0, virtual_height - menu_height);

        return;
    }

    if (menu_ != menu::none && cursor_free_ && click &&
        inside(mx, my, {menu_x_, menu_y_, menu_width, 52}))
    {
        dragging_menu_ = true;
        drag_x_ = mx - menu_x_;
        drag_y_ = my - menu_y_;

        return;
    }

    if (click && cursor_free_ &&
        (menu_ == menu::none ||
         !inside(mx, my, menu_rect(menu_x_, menu_y_))))
    {
        for (auto it = labels_.rbegin(); it != labels_.rend(); ++it)
        {
            if (!it->visible || !it->object) continue;
            if (!inside(mx, my, label_rect(*it, width, height))) continue;
            selected_object_ = it->object;
            menu_ = menu::properties;
            delete_armed_ = false;
            selected_row_ = scroll_first_ = 0;

            break;
        }
    }
}

void session::add(std::string id, std::string name,
                  std::function<void()> insert, std::string group)
{
    if (id.empty() || !insert)
        throw std::invalid_argument("Editor insert requires an identifier and callback");
    std::lock_guard<std::mutex> guard(mutex_);

    auto found = std::find_if(catalog_.begin(), catalog_.end(),
                              [&](const object_type &item)
                              { return item.id == id; });
    object_type item{id, name, std::move(group)};

    if (found != catalog_.end()) *found = std::move(item);
    else
        catalog_.push_back(std::move(item));
    insert_handlers_[std::move(id)] = std::move(insert);
}

void session::on_edit(std::string property_id,
                      std::function<void(std::uint64_t, double)> edit)
{
    std::lock_guard<std::mutex> guard(mutex_);
    edit_handlers_[std::move(property_id)] = std::move(edit);
}

void session::on_remove(std::function<void(std::uint64_t)> remove)
{
    std::lock_guard<std::mutex> guard(mutex_);
    remove_handler_ = std::move(remove);
}

void session::dispatch()
{
    for (const auto &action : take_actions())
    {
        std::function<void()> insert;
        std::function<void(std::uint64_t, double)> edit;
        std::function<void(double)> bound;
        std::function<void(std::uint64_t)> remove;
        {
            std::lock_guard<std::mutex> guard(mutex_);

            if (action.kind == action_kind::insert) {
                auto it = insert_handlers_.find(action.type);
                if (it != insert_handlers_.end()) insert = it->second;
            }
            else if (action.kind == action_kind::edit) {
                auto object = bound_properties_.find(action.object);

                if (object != bound_properties_.end()) {
                    auto it = object->second.find(action.property_id);
                    if (it != object->second.end()) bound = it->second;
                }

                if (!bound) {
                    auto it = edit_handlers_.find(action.property_id);
                    if (it != edit_handlers_.end()) edit = it->second;
                }
            }
            else
                remove = remove_handler_;
        }

        if (insert) insert();
        if (bound) bound(action.value);
        if (edit) edit(action.object, action.value);
        if (remove) remove(action.object);
    }
}

std::vector<session::action> session::take_actions()
{
    std::lock_guard<std::mutex> guard(mutex_);
    std::vector<action> out;
    out.swap(pending_);

    return out;
}

draw_list session::paint(int width, int height) const
{
    std::lock_guard<std::mutex> guard(mutex_);

    draw_list out;

    if (!visible_ || visualization_ || width <= 0 || height <= 0) return out;

    // Use the same topmost label and panel occlusion rules as clicking.
    std::uint64_t hovered_object = 0;

    if (cursor_free_ && !dragging_menu_ && mouse_x_ >= 0 && mouse_y_ >= 0 &&
        (menu_ == menu::none || !inside(mouse_x_, mouse_y_, menu_rect(menu_x_, menu_y_))))
        for (auto it = labels_.rbegin(); it != labels_.rend(); ++it)
            if (it->visible && it->object && inside(mouse_x_, mouse_y_, label_rect(*it, width, height))) {
                hovered_object = it->object;

                break;
            }

    // World icons appear whenever '?' mode is enabled, even without TAB.
    for (const auto &label : labels_)
    {
        if (!label.visible || label.object == 0) continue;

        const rect r = label_rect(label, width, height);
        const bool selected = menu_ == menu::properties && label.object == selected_object_;
        const bool highlighted = selected || label.object == hovered_object;
        // All world labels use the same blank shaded plate and font.
        // Previously LIGHT/CUBE contained baked lettering while other
        // labels used the CP437 font, so their appearance never matched.
        image(out, draw_command::texture::label_plate, r);

        if (highlighted) out.back().tint = {1.f, .25f, .25f, 1.f};
        text(out, r.x + r.w / 2, r.y + 3, label.text,
             draw_command::alignment::center);
    }

    // FPS/frame time on the right, separate from Hell's floating menus.
    if (frame_time_ms_ > 0) {
        std::ostringstream fps, ms;
        fps << std::fixed << std::setprecision(1)
            << 1000.0 / frame_time_ms_ << " FPS";
        ms << std::fixed << std::setprecision(2)
           << frame_time_ms_ << " MS";
        text(out, virtual_width - 16, 12, fps.str(),
             draw_command::alignment::right);
        text(out, virtual_width - 16, 28, ms.str(),
             draw_command::alignment::right);
    }

    if (menu_ == menu::none) return out;

    // Recreate Editor::UpdateRenderItems from Hell2024 @7362a3a:
    // textured MenuBG and four border pieces + four exact corner pieces.
    const rect area = menu_rect(menu_x_, menu_y_);
    using tex = draw_command::texture;
    image(out, tex::menu_bg, area);
    image(out, tex::border_horizontal, {area.x, area.y, area.w, 3});
    image(out, tex::border_horizontal,
          {area.x, area.y + area.h - 3, area.w, 3});
    image(out, tex::border_vertical, {area.x, area.y, 3, area.h});
    image(out, tex::border_vertical,
          {area.x + area.w - 3, area.y, 3, area.h});
    image(out, tex::corner_tl, {area.x, area.y, 4, 4});
    image(out, tex::corner_tr, {area.x + area.w - 4, area.y, 4, 4});
    image(out, tex::corner_bl, {area.x, area.y + area.h - 4, 4, 4});
    image(out, tex::corner_br,
          {area.x + area.w - 4, area.y + area.h - 4, 4, 4});

    std::string heading = menu_ == menu::insert ? "------ INSERT ------" : "------ SELECT ------";

    if (menu_ == menu::properties)
        for (const auto &label : labels_)
            if (label.object == selected_object_ && !label.text.empty()) {
                heading = "------ " + label.text + " ------";

                break;
            }

    text(out, area.x + area.w / 2, area.y + margin_y + 4,
         heading, draw_command::alignment::center);

    const std::vector<property> *values = nullptr;

    if (menu_ == menu::properties) {
        const auto it = properties_.find(selected_object_);
        if (it != properties_.end()) values = &it->second;
    }

    const int rows = menu_ == menu::insert ? int(catalog_.size()) : values ? int(values->size())
                                                                           : 0;

    const bool inspector = menu_ == menu::properties;

    if (rows == 0 && !inspector) {
        text(out, area.x + margin_x, area.y + 70, "NO TYPES REGISTERED");

        return out;
    }

    // TextBlitter uses glyph-specific advances and 16px line height.
    // Wrap long host-defined strings by actual glyph widths.
    const auto measure = [&](const std::string &str)
    {
        int total = 0;

        for (unsigned char c : str)
            total += c == ' ' ? 6 : font_widths_[c < 128 ? c : 0];

        return total;
    };

    const auto wrap = [&](const std::string &str)
    {
        std::vector<std::string> lines;
        std::string current;

        const int limit = menu_width - 2 * margin_x;

        for (char c : str)
        {
            if (c == '\n') {
                lines.push_back(current);
                current.clear();

                continue;
            }

            if (measure(current + std::string(1, c)) > limit &&
                !current.empty())
            {
                lines.push_back(current);
                current.clear();
            }

            if (c != ' ' || !current.empty()) current += c;
        }

        lines.push_back(current);

        return lines;
    };

    std::vector<std::vector<std::string>> lines;
    std::vector<int> heights;
    lines.reserve(std::size_t(rows));
    heights.reserve(std::size_t(rows));

    for (int i = 0; i < rows; ++i)
    {
        const bool selected = i == selected_row_;

        std::string value;

        if (menu_ == menu::insert)
            value = (selected ? "  >" : "   ") +
                    catalog_[std::size_t(i)].name;
        else {
            const auto &p = (*values)[std::size_t(i)];
            value = "  " + p.name + ": " +
                    (selected ? "<" : " ") + value_string(p) + (selected ? ">" : " ");
        }

        lines.push_back(wrap(value));
        heights.push_back(std::max(1, int(lines.back().size())) *
                          line_height);
    }

    const int start_y = area.y + 56;
    // Keep Delete fixed at the bottom, independent of the scrollable
    // property list (which may be longer than the menu).
    const int bottom = area.y + area.h - margin_y -
                       (inspector ? line_height + 8 : 0);

    int first = std::clamp(scroll_first_, 0, std::max(0, rows - 1));

    if (selected_row_ < first) first = selected_row_;
    // Selection of Delete does not scroll away the earlier fields.
    const int selected_for_scroll = inspector ? std::min(selected_row_, std::max(0, rows - 1)) : selected_row_;
    const auto is_visible = [&](int initial)
    {
        int pos = start_y;

        for (int i = initial; i <= selected_for_scroll && i < rows; ++i)
        {
            if (pos + heights[std::size_t(i)] > bottom) return false;
            pos += heights[std::size_t(i)];
        }

        return true;
    };

    while (first < selected_for_scroll && !is_visible(first))
        ++first;
    scroll_first_ = first;

    int y = start_y;

    for (int row = first; row < rows; ++row)
    {
        if (y + line_height > bottom) break;

        for (const auto &line : lines[std::size_t(row)])
        {
            if (y + line_height > bottom) break;
            text(out, area.x + margin_x, y, line);
            y += line_height;
        }
    }

    if (inspector) {
        const bool selected = selected_row_ == rows;
        const auto row = std::string("  Delete: ") +
                         (selected ? "<" : " ") +
                         (delete_armed_ ? "TRUE" : "FALSE") + (selected ? ">" : " ");

        for (const auto &part : wrap(row))
        {
            // Delete always occupies its reserved bottom line.
            text(out, area.x + margin_x,
                 area.y + area.h - margin_y - line_height, part);
            break;
        }
    }

    return out;
}

bool session::visible() const
{
    std::lock_guard<std::mutex> guard(mutex_);

    return visible_;
}

bool session::cursor_free() const
{
    std::lock_guard<std::mutex> guard(mutex_);

    return cursor_free_;
}

bool session::captures_input() const
{
    std::lock_guard<std::mutex> guard(mutex_);

    return cursor_free_ || (visible_ && menu_ != menu::none);
}

std::uint32_t session::visualization() const
{
    std::lock_guard<std::mutex> guard(mutex_);

    return visible_ ? visualization_ : 0u;
}
} // namespace vs::editor
