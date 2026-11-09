#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace editor
{
    // Input is supplied by the host's window system. Values are one-frame
    // press events, not continuous key state. Key meanings are fixed here.
    struct input
    {
        bool question=false,tab=false,enter=false,escape=false;
        bool up=false,down=false,left=false,right=false,click=false;
        bool shift=false; // Hold Shift to change numeric properties by 10 steps.
        bool mouse_down=false; // Held button for dragging floating panels
        int mouse_x=0,mouse_y=0;
        int viewport_width=0,viewport_height=0; // Framebuffer pixel coordinates
    };

    struct object_type
    {
        std::string id;       // Stable host-defined placement identifier.
        std::string name;     // Visible entry in the Insert menu.
        std::string group;    // Optional category (ordering is preserved).
    };

    struct screen_label
    {
        std::uint64_t object=0;
        std::string text;
        int x=0,y=0;          // Projected screen position; host owns projection.
        bool visible=true;
    };

    enum class property_kind { real,integer,boolean,choice };
    struct property
    {
        std::string id;
        std::string name;
        property_kind kind=property_kind::real;
        double value=0,minimum=0,maximum=100,step=1;
        std::vector<std::string> choices;
        int precision=2; // Exact Hell2024-style numeric precision, host-defined.
    };

    enum class action_kind { insert,edit,remove };
    struct action
    {
        action_kind kind=action_kind::insert;
        std::string type;
        std::uint64_t object=0;
        std::string property_id;
        double value=0;
    };

    struct color { float r=1,g=1,b=1,a=1; };
    struct draw_command
    {
        // Geometry matches Hell2024's 960x540 PRESENT_WIDTH/HEIGHT canvas.
        enum class type { rectangle,text,image } kind=type::rectangle;
        enum class texture { menu_bg,border_horizontal,border_vertical,corner_tl,corner_tr,corner_bl,corner_br,light_icon,cube_icon } image=texture::menu_bg;
        enum class alignment { left,center,right } text_alignment=alignment::left;
        int x=0,y=0,width=0,height=0,scale=1;
        color tint{};
        std::string text;
    };
    using draw_list=std::vector<draw_command>;

    // No engine, CSG, GLFW, scene or graphics ownership. Host publishes
    // projected objects/properties and consumes actions on its main thread.
    // Safe to call paint() from a separate render thread.
    class session
    {
    public:
        void set_catalog(std::vector<object_type> types);
        void set_labels(std::vector<screen_label> labels);
        void set_properties(std::uint64_t object,std::vector<property> values);
        void handle(const input& events);
        // Host reports CPU frame presentation interval (not GPU pass time).
        void set_frame_time(double milliseconds);
        // Original TextBlitter character advances, supplied once by asset loader.
        void set_font_metrics(std::array<int,128> widths,int line_height=16);
        std::vector<action> take_actions();
        draw_list paint(int width,int height) const;
        bool visible() const;
        bool cursor_free() const;
        bool captures_input() const;
    private:
        enum class menu { none,insert,properties };
        mutable std::mutex mutex_;
        bool visible_=false,cursor_free_=false;
        menu menu_=menu::none;
        int selected_row_=0;
        mutable int scroll_first_=0;
        double frame_time_ms_=0;
        std::array<int,128> font_widths_{};
        int line_height_=16;
        int menu_x_=60,menu_y_=60;
        bool dragging_menu_=false;
        bool delete_armed_=false; // Committed only on Escape from Properties.
        int drag_x_=0,drag_y_=0;

        int mouse_x_=-1,mouse_y_=-1; // Latest cursor on the virtual canvas.
        std::uint64_t selected_object_=0;
        std::vector<object_type> catalog_;
        std::vector<screen_label> labels_;
        std::unordered_map<std::uint64_t,std::vector<property>> properties_;
        std::vector<action> pending_;
    };

    // Convenience instance for single-window applications. Other hosts can
    // construct independent sessions for multiple views/windows.
    session& instance();
}
