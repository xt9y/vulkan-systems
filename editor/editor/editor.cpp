#include "editor.hpp"
#include <cmath>
#include <iomanip>
#include <sstream>
#include <utility>

namespace editor
{
    namespace
    {
        // Original Hell2024: PRESENT_WIDTH=960, PRESENT_HEIGHT=540.
        // Menu uses g_menuLocation=(76,460), margins=(16,20) and 200px width.
        // Game host controls all object types/labels/properties.
        constexpr int virtual_width=960,virtual_height=540;
        constexpr int menu_width=184,menu_height=222;
        constexpr int margin_x=16,margin_y=20;
        constexpr int line_height=16;
                constexpr color white{0.80f,0.80f,0.83f,1};
        constexpr color teal{0.26f,0.46f,0.44f,1};
        constexpr color red{0.55f,0.12f,0.15f,1};

        struct rect { int x=0,y=0,w=0,h=0; };
        bool inside(int x,int y,rect r)
        {
            return x>=r.x && y>=r.y && x<r.x+r.w && y<r.y+r.h;
        }
        int virtual_x(int x,int viewport)
        {
            return std::clamp(int(double(x)*virtual_width/
                       double(std::max(1,viewport))),0,virtual_width);
        }
        int virtual_y(int y,int viewport)
        {
            return std::clamp(int(double(y)*virtual_height/
                       double(std::max(1,viewport))),0,virtual_height);
        }
        void rectangle(draw_list& out,rect area,color tint)
        {
            if(area.w<=0 || area.h<=0)return;
            draw_command item{};
            item.kind=draw_command::type::rectangle;
            item.x=area.x;item.y=area.y;
            item.width=area.w;item.height=area.h;item.tint=tint;
            out.push_back(std::move(item));
        }
        void image(draw_list& out,draw_command::texture texture,rect area)
        {
            if(area.w<=0 || area.h<=0)return;
            draw_command item{};
            item.kind=draw_command::type::image;
            item.image=texture;
            item.x=area.x;item.y=area.y;
            item.width=area.w;item.height=area.h;
            out.push_back(std::move(item));
        }
        void text(draw_list& out,int x,int y,std::string content,
                  draw_command::alignment align=draw_command::alignment::left)
        {
            draw_command item{};
            item.kind=draw_command::type::text;
            item.x=x;item.y=y;item.scale=1;
            item.text_alignment=align;
            item.text=std::move(content);
            item.tint=white;
            out.push_back(std::move(item));
        }
        rect menu_rect(int x,int y)
        {
            return {x,y,menu_width,menu_height};
        }
        rect label_rect(const screen_label& label,int width,int height)
        {
            const int cx=virtual_x(label.x,width);
            const int cy=virtual_y(label.y,height);
            const int label_width=std::max(62,
                int(label.text.size())*8+12);
            constexpr int label_height=22;
            // All textured world signs share the same centered anchor.
            const int top=cy-label_height/2;
            return {std::clamp(cx-label_width/2,0,
                     std::max(0,virtual_width-label_width)),
                    std::clamp(top,0,virtual_height-label_height),
                    label_width,label_height};
        }
        std::string value_string(const property& p)
        {
            if(p.kind==property_kind::boolean)
                return p.value!=0 ? "TRUE":"FALSE";
            if(p.kind==property_kind::choice)
            {
                const auto index=int(std::round(p.value));
                return index>=0 && index<int(p.choices.size())
                    ?p.choices[std::size_t(index)]:"NONE";
            }
            std::ostringstream out;
            if(p.kind==property_kind::integer)
                out<<std::llround(p.value);
            else
                out<<std::fixed<<std::setprecision(
                    std::clamp(p.precision,0,6))<<p.value;
            return out.str();
        }
    }

    session& instance()
    {
        static session value;
        return value;
    }

    void session::set_catalog(std::vector<object_type> types)
    {
        std::lock_guard<std::mutex> guard(mutex_);
        catalog_=std::move(types);
        if(menu_==menu::insert)
            selected_row_=std::clamp(selected_row_,0,
                std::max(0,int(catalog_.size())-1));
    }

    void session::set_labels(std::vector<screen_label> labels)
    {
        std::lock_guard<std::mutex> guard(mutex_);
        labels_=std::move(labels);
    }

    void session::set_properties(std::uint64_t object,std::vector<property> values)
    {
        std::lock_guard<std::mutex> guard(mutex_);
        if(auto edits=local_edits_.find(object);edits!=local_edits_.end())
        {
            const auto now=std::chrono::steady_clock::now();
            for(auto& p:values)
            {
                const auto it=edits->second.find(p.id);
                if(it==edits->second.end())continue;
                const double tolerance=std::max(1e-4,p.step*.05);
                if(std::abs(p.value-it->second.value)<=tolerance ||
                   now-it->second.updated>std::chrono::seconds(3))
                    edits->second.erase(it);
                else
                    p.value=it->second.value;
            }
            if(edits->second.empty())local_edits_.erase(edits);
        }
        properties_[object]=std::move(values);
        if(menu_==menu::properties && selected_object_==object)
            selected_row_=std::clamp(selected_row_,0,
                int(properties_[object].size()));
    }

    void session::reset()
    {
        std::lock_guard<std::mutex> guard(mutex_);
        visible_=cursor_free_=dragging_menu_=delete_armed_=false;
        menu_=menu::none;
        selected_row_=scroll_first_=0;
        selected_object_=0;
        labels_.clear();
        properties_.clear();
        local_edits_.clear();
        pending_.clear();
    }

    void session::set_font_metrics(std::array<int,128> widths,int line_height)
    {
        std::lock_guard<std::mutex> guard(mutex_);
        font_widths_=std::move(widths);
        line_height_=std::max(1,line_height);
    }

    void session::set_frame_time(double milliseconds)
    {
        if(!std::isfinite(milliseconds)||milliseconds<=0)return;
        std::lock_guard<std::mutex> guard(mutex_);
        frame_time_ms_=frame_time_ms_>0 ?
            frame_time_ms_*0.85+milliseconds*0.15:milliseconds;
    }

    void session::handle(const input& events)
    {
        std::lock_guard<std::mutex> guard(mutex_);
        // Controls are intentionally hardcoded here, not in VulkanGame.
        if(events.tab)cursor_free_=!cursor_free_;
        const bool cursor_inside=events.viewport_width>0 && events.viewport_height>0 &&
            events.mouse_x>=0 && events.mouse_y>=0 &&
            events.mouse_x<events.viewport_width && events.mouse_y<events.viewport_height;
        mouse_x_=cursor_inside?virtual_x(events.mouse_x,events.viewport_width):-1;
        mouse_y_=cursor_inside?virtual_y(events.mouse_y,events.viewport_height):-1;
        if(events.question)
        {
            visible_=!visible_;
            if(!visible_)cursor_free_=false;
            menu_=menu::none;
            selected_object_=0;
            dragging_menu_=false;
            delete_armed_=false;
            selected_row_=scroll_first_=0;
        }
        if(!visible_)return;
        if(events.escape)
        {
            // Deletion is deliberately deferred until closing the selected
            // object's inspector; simply toggling TRUE changes no world state.
            if(menu_==menu::properties && delete_armed_ && selected_object_)
                pending_.push_back({action_kind::remove,"",
                                    selected_object_,"",0});
            menu_=menu::none;
            selected_object_=0;
            dragging_menu_=false;
            delete_armed_=false;
            return;
        }

        const int mx=virtual_x(events.mouse_x,events.viewport_width);
        const int my=virtual_y(events.mouse_y,events.viewport_height);
        if(!events.mouse_down)dragging_menu_=false;
        if(dragging_menu_ && events.mouse_down)
        {
            menu_x_=std::clamp(mx-drag_x_,0,virtual_width-menu_width);
            menu_y_=std::clamp(my-drag_y_,0,virtual_height-menu_height);
            return;
        }
        if(menu_!=menu::none && cursor_free_ && events.click &&
           inside(mx,my,{menu_x_,menu_y_,menu_width,52}))
        {
            dragging_menu_=true;
            drag_x_=mx-menu_x_;
            drag_y_=my-menu_y_;
            return;
        }

        if(menu_==menu::insert)
        {
            const int jump=events.shift?10:1;
            if(events.up)selected_row_=std::max(0,selected_row_-jump);
            if(events.down)selected_row_=std::min(
                std::max(0,int(catalog_.size())-1),selected_row_+jump);
            if(events.enter && !catalog_.empty())
            {
                pending_.push_back({action_kind::insert,
                    catalog_[std::size_t(selected_row_)].id,0,"",0});
                menu_=menu::none;
                scroll_first_=0;
            }
        }
        else if(menu_==menu::properties)
        {
            const auto it=properties_.find(selected_object_);
            const int property_count=it!=properties_.end() ?
                int(it->second.size()):0;
            // The Delete field is always the last inspector entry. It is
            // built into VulkanEditor, not part of host property schemas.
            const int jump=events.shift?10:1;
            if(events.up)selected_row_=std::max(0,selected_row_-jump);
            if(events.down)selected_row_=std::min(
                property_count,selected_row_+jump);
            if(events.left || events.right)
            {
                if(selected_row_==property_count)
                {
                    if(events.right)delete_armed_=true;
                    if(events.left)delete_armed_=false;
                }
                else if(it!=properties_.end())
                {
                    property& p=it->second[std::size_t(selected_row_)];
                    const double before=p.value;
                    if(p.kind==property_kind::boolean)
                        p.value=events.right?1.0:0.0;
                    else
                    {
                        const double upper=p.kind==property_kind::choice ?
                            double(std::max(0,int(p.choices.size())-1)):
                            p.maximum;
                        // Right/Left change one host-defined unit, or ten
                        // with Shift. Colors may use e.g. percentage units.
                        const double unit=p.step>0?p.step:1.0;
                        const double delta=unit*(events.command?100.0:
                                                  events.shift?10.0:1.0);
                        p.value=std::clamp(before+
                            (events.right?delta:-delta),
                            std::min(p.minimum,upper),
                            std::max(p.minimum,upper));
                        if(p.kind==property_kind::integer ||
                           p.kind==property_kind::choice)
                            p.value=std::round(p.value);
                    }
                    if(p.value!=before)
                    {
                        local_edits_[selected_object_][p.id]={
                            p.value,std::chrono::steady_clock::now()};
                        pending_.push_back({action_kind::edit,"",
                            selected_object_,p.id,p.value});
                    }
                }
            }
        }
        else if(events.enter)
        {
            menu_=menu::insert;
            selected_row_=scroll_first_=0;
        }

        // Original editor selection is by interactive world icons; the host
        // projects its own objects and supplies stable IDs.
        if(events.click && cursor_free_ &&
           (menu_==menu::none ||
            !inside(mx,my,menu_rect(menu_x_,menu_y_))))
        {
            for(auto it=labels_.rbegin();it!=labels_.rend();++it)
            {
                if(!it->visible || it->object==0)continue;
                if(!inside(mx,my,label_rect(*it,
                    events.viewport_width,events.viewport_height)))continue;
                selected_object_=it->object;
                menu_=menu::properties;
                delete_armed_=false;
                selected_row_=scroll_first_=0;
                break;
            }
        }
    }

    std::vector<action> session::take_actions()
    {
        std::lock_guard<std::mutex> guard(mutex_);
        std::vector<action> out;
        out.swap(pending_);
        return out;
    }

    draw_list session::paint(int width,int height) const
    {
        std::lock_guard<std::mutex> guard(mutex_);
        draw_list out;
        if(!visible_||width<=0||height<=0)return out;

        // Use the same topmost label and panel occlusion rules as clicking.
        std::uint64_t hovered_object=0;
        if(cursor_free_ && !dragging_menu_ && mouse_x_>=0 && mouse_y_>=0 &&
           (menu_==menu::none || !inside(mouse_x_,mouse_y_,menu_rect(menu_x_,menu_y_))))
            for(auto it=labels_.rbegin();it!=labels_.rend();++it)
                if(it->visible && it->object && inside(mouse_x_,mouse_y_,label_rect(*it,width,height)))
                {
                    hovered_object=it->object;
                    break;
                }

        // World icons appear whenever '?' mode is enabled, even without TAB.
        for(const auto& label:labels_)
        {
            if(!label.visible || label.object==0)continue;
            const rect r=label_rect(label,width,height);
            const bool selected=menu_==menu::properties && label.object==selected_object_;
            const bool highlighted=selected || label.object==hovered_object;
            const auto sprite=label.text=="LIGHT"?
                draw_command::texture::light_icon:label.text=="CUBE"?
                draw_command::texture::cube_icon:
                draw_command::texture::label_plate;
            image(out,sprite,r);
            if(highlighted)out.back().tint={1.f,.25f,.25f,1.f};
            if(sprite==draw_command::texture::label_plate)
                text(out,r.x+r.w/2,r.y+3,label.text,
                     draw_command::alignment::center);
        }

        // FPS/frame time on the right, separate from Hell's floating menus.
        if(frame_time_ms_>0)
        {
            std::ostringstream fps,ms;
            fps<<std::fixed<<std::setprecision(1)
               <<1000.0/frame_time_ms_<<" FPS";
            ms<<std::fixed<<std::setprecision(2)
              <<frame_time_ms_<<" MS";
            text(out,virtual_width-16,12,fps.str(),
                draw_command::alignment::right);
            text(out,virtual_width-16,28,ms.str(),
                draw_command::alignment::right);
        }

        if(menu_==menu::none)return out;

        // Recreate Editor::UpdateRenderItems from Hell2024 @7362a3a:
        // textured MenuBG and four border pieces + four exact corner pieces.
        const rect area=menu_rect(menu_x_,menu_y_);
        using tex=draw_command::texture;
        image(out,tex::menu_bg,area);
        image(out,tex::border_horizontal,{area.x,area.y,area.w,3});
        image(out,tex::border_horizontal,
              {area.x,area.y+area.h-3,area.w,3});
        image(out,tex::border_vertical,{area.x,area.y,3,area.h});
        image(out,tex::border_vertical,
              {area.x+area.w-3,area.y,3,area.h});
        image(out,tex::corner_tl,{area.x,area.y,4,4});
        image(out,tex::corner_tr,{area.x+area.w-4,area.y,4,4});
        image(out,tex::corner_bl,{area.x,area.y+area.h-4,4,4});
        image(out,tex::corner_br,
              {area.x+area.w-4,area.y+area.h-4,4,4});

        std::string heading=menu_==menu::insert?"------ INSERT ------":"------ SELECT ------";
        if(menu_==menu::properties)
            for(const auto& label:labels_)
                if(label.object==selected_object_ && !label.text.empty())
                {
                    heading="------ "+label.text+" ------";
                    break;
                }
        text(out,area.x+area.w/2,area.y+margin_y+4,
             heading,draw_command::alignment::center);

        const std::vector<property>* values=nullptr;
        if(menu_==menu::properties)
        {
            const auto it=properties_.find(selected_object_);
            if(it!=properties_.end())values=&it->second;
        }
        const int rows=menu_==menu::insert?int(catalog_.size()):
            values?int(values->size()):0;
        const bool inspector=menu_==menu::properties;
        if(rows==0 && !inspector)
        {
            text(out,area.x+margin_x,area.y+70,"NO TYPES REGISTERED");
            return out;
        }

        // TextBlitter uses glyph-specific advances and 16px line height.
        // Wrap long host-defined strings by actual glyph widths.
        const auto measure=[&](const std::string& str)
        {
            int total=0;
            for(unsigned char c:str)
                total+=c==' '?6:font_widths_[c<128?c:0];
            return total;
        };
        const auto wrap=[&](const std::string& str)
        {
            std::vector<std::string> lines;
            std::string current;
            const int limit=menu_width-2*margin_x;
            for(char c:str)
            {
                if(c=='\n')
                {
                    lines.push_back(current);
                    current.clear();
                    continue;
                }
                if(measure(current+std::string(1,c))>limit &&
                   !current.empty())
                {
                    lines.push_back(current);
                    current.clear();
                }
                if(c!=' '||!current.empty())current+=c;
            }
            lines.push_back(current);
            return lines;
        };

        std::vector<std::vector<std::string>> lines;
        std::vector<int> heights;
        lines.reserve(std::size_t(rows));
        heights.reserve(std::size_t(rows));
        for(int i=0;i<rows;++i)
        {
            const bool selected=i==selected_row_;
            std::string value;
            if(menu_==menu::insert)
                value=(selected?"  >":"   ")+
                    catalog_[std::size_t(i)].name;
            else
            {
                const auto& p=(*values)[std::size_t(i)];
                value="  "+p.name+": "+
                    (selected?"<":" ")+value_string(p)+(selected?">":" ");
            }
            lines.push_back(wrap(value));
            heights.push_back(std::max(1,int(lines.back().size()))*
                line_height);
        }

        const int start_y=area.y+56;
        // Keep Delete fixed at the bottom, independent of the scrollable
        // property list (which may be longer than the menu).
        const int bottom=area.y+area.h-margin_y-
            (inspector?line_height+8:0);
        int first=std::clamp(scroll_first_,0,std::max(0,rows-1));
        if(selected_row_<first)first=selected_row_;
        // Selection of Delete does not scroll away the earlier fields.
        const int selected_for_scroll=inspector ?
            std::min(selected_row_,std::max(0,rows-1)):
            selected_row_;
        const auto is_visible=[&](int initial)
        {
            int pos=start_y;
            for(int i=initial;i<=selected_for_scroll && i<rows;++i)
            {
                if(pos+heights[std::size_t(i)]>bottom)return false;
                pos+=heights[std::size_t(i)];
            }
            return true;
        };
        while(first<selected_for_scroll && !is_visible(first))++first;
        scroll_first_=first;
        int y=start_y;
        for(int row=first;row<rows;++row)
        {
            if(y+line_height>bottom)break;
            for(const auto& line:lines[std::size_t(row)])
            {
                if(y+line_height>bottom)break;
                text(out,area.x+margin_x,y,line);
                y+=line_height;
            }
        }
        if(inspector)
        {
            const bool selected=selected_row_==rows;
            const auto row=std::string("  Delete: ")+
                (selected?"<":" ")+
                (delete_armed_?"TRUE":"FALSE")+(selected?">":" ");
            for(const auto& part:wrap(row))
            {
                // Delete always occupies its reserved bottom line.
                text(out,area.x+margin_x,
                     area.y+area.h-margin_y-line_height,part);
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
        return cursor_free_ || (visible_&&menu_!=menu::none);
    }
}
