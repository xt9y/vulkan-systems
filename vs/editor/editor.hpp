#pragma once
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace vs::editor
{
// Intentional commands, not a frame-long packet of key booleans.
// Applications can forward these from any window/input backend.
enum class key
{
    toggle,
    cursor,
    confirm,
    cancel,
    up,
    down,
    left,
    right,
    erase,
    view_1,
    view_2,
    view_3,
    view_4,
    view_5,
    view_6,
    view_7,
    view_8,
    view_9
};

struct screen_label
{
    std::uint64_t object = 0;
    std::string text;

    int x = 0, y = 0; // Projected screen position; host owns projection.
    bool visible = true;
};

enum class property_kind
{
    real,
    integer,
    boolean,
    choice
};

struct property
{
    std::string id;
    std::string name;

    property_kind kind = property_kind::real;
    double value = 0, minimum = 0, maximum = 100, step = 1;

    std::vector<std::string> choices;

    int precision = 2; // Exact Hell2024-style numeric precision, host-defined.
};

struct color
{ float r = 1, g = 1, b = 1, a = 1; };
struct draw_command
{
    // Geometry matches Hell2024's 960x540 PRESENT_WIDTH/HEIGHT canvas.
    enum class type
    {
        rectangle,
        text,
        image
    } kind = type::rectangle;

    enum class texture
    {
        menu_bg,
        border_horizontal,
        border_vertical,
        corner_tl,
        corner_tr,
        corner_bl,
        corner_br,
        label_plate
    } image = texture::menu_bg;

    enum class alignment
    {
        left,
        center,
        right
    } text_alignment = alignment::left;

    int x = 0, y = 0, width = 0, height = 0, scale = 1;

    color tint{};
    std::string text;
};

using draw_list = std::vector<draw_command>;

// No engine, CSG, GLFW, scene or graphics ownership. Host publishes
// projected objects/properties and consumes actions on its main thread.
// Safe to call paint() from a separate render thread.
class session
{
public:
    // Register behavior once; no hand-built bool input packets or action switches.
    void add(std::string id, std::string name, std::function<void()> insert,
             std::string group = {});
    void on_edit(std::string property_id,
                 std::function<void(std::uint64_t, double)> edit);
    void on_remove(std::function<void(std::uint64_t)> remove);
    void set_labels(std::vector<screen_label> labels);
    void set_properties(std::uint64_t object, std::vector<property> values);
    // Data and change behavior together; VS never owns the project object.
    void bind(std::uint64_t object, property description,
              std::function<void(double)> apply);
    void reset();
    void press(key command, bool shift = false, bool accelerated = false);
    void pointer(int x, int y, int width, int height, bool down = false);
    void dispatch(); // invokes registered behavior outside the session lock
    // Host reports CPU frame presentation interval (not GPU pass time).
    void set_frame_time(double milliseconds);
    // Original TextBlitter character advances, supplied once by asset loader.
    void set_font_metrics(std::array<int, 128> widths, int line_height = 16);
    draw_list paint(int width, int height) const;
    bool visible() const;
    bool cursor_free() const;
    bool captures_input() const;
    // 0 = ordinary editor, 1..9 = exclusive inspection layer.
    // An inspection view exists only while '?' editor mode is visible.
    std::uint32_t visualization() const;

private:
    struct object_type
    {
        std::string id;    // Stable host-defined placement identifier.
        std::string name;  // Visible entry in the Insert menu.
        std::string group; // Optional category (ordering is preserved).
    };

    enum class action_kind
    {
        insert,
        edit,
        remove
    };

    struct action
    {
        action_kind kind = action_kind::insert;

        std::string type;
        std::uint64_t object = 0;
        std::string property_id;

        double value = 0;
    };

    std::vector<action> take_actions();
    enum class menu
    {
        none,
        insert,
        properties
    };

    mutable std::mutex mutex_;

    bool visible_ = false, cursor_free_ = false;

    std::uint32_t visualization_ = 0;

    menu menu_ = menu::none;
    int selected_row_ = 0;

    mutable int scroll_first_ = 0;

    double frame_time_ms_ = 0;

    std::array<int, 128> font_widths_{};

    int line_height_ = 16;
    int menu_x_ = 60, menu_y_ = 60;
    bool dragging_menu_ = false, pointer_down_ = false;
    bool delete_armed_ = false; // Committed only on Escape from Properties.
    int drag_x_ = 0, drag_y_ = 0;

    int mouse_x_ = -1, mouse_y_ = -1; // Latest cursor on the virtual canvas.
    std::uint64_t selected_object_ = 0;
    std::vector<object_type> catalog_;
    std::vector<screen_label> labels_;
    std::unordered_map<std::uint64_t, std::vector<property>> properties_;
    struct local_edit
    {
        double value = 0;

        std::chrono::steady_clock::time_point updated{};
    };

    std::unordered_map<std::uint64_t, std::unordered_map<std::string, local_edit>> local_edits_;
    std::vector<action> pending_;
    std::unordered_map<std::string, std::function<void()>> insert_handlers_;
    std::unordered_map<std::string,
                       std::function<void(std::uint64_t, double)>>
        edit_handlers_;
    std::unordered_map<std::uint64_t,
                       std::unordered_map<std::string, std::function<void(double)>>>
        bound_properties_;
    std::function<void(std::uint64_t)> remove_handler_;
};

} // namespace vs::editor
