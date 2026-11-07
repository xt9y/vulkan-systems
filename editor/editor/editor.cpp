#include "editor.hpp"
#include <cmath>
#include <iomanip>
#include <sstream>
#include <utility>

namespace editor
{
    namespace
    {
        constexpr color background{0.035f,0.027f,0.071f,1};
        constexpr color frame{0.72f,0.71f,0.77f,1};
        constexpr color foreground{0.90f,0.89f,0.93f,1};
        constexpr color highlight{0.23f,0.19f,0.32f,1};
        constexpr color label_color{0.20f,0.37f,0.35f,1};
        constexpr color selected_label_color{0.45f,0.12f,0.12f,1};
        // Fixed palette/layout deliberately mirrors the old Rogue editor.
        constexpr int line_height=24;
        constexpr int text_scale=2;
        constexpr int side_margin=14;
        constexpr int panel_width_limit=464;
        constexpr int panel_padding=34;
        constexpr int header_height=100;

        struct rectangle { int x,y,w,h; };
        rectangle label_rect(const screen_label& label,int width,int height)
        {
            const int w=std::max(66,int(label.text.size())*12+20);
            const int x=std::clamp(label.x-w/2,0,std::max(0,width-w));
            const int y=std::clamp(label.y-32,0,std::max(0,height-30));
            return {x,y,w,30};
        }
        void solid(draw_list& out,rectangle r,color c)
        {
            if(r.w<=0 || r.h<=0)return;
            draw_command item{};
            item.x=r.x;item.y=r.y;item.width=r.w;item.height=r.h;
            item.tint=c;
            out.push_back(std::move(item));
        }
        void border(draw_list& out,rectangle r,color c)
        {
            solid(out,{r.x,r.y,r.w,2},c);
            solid(out,{r.x,r.y+r.h-2,r.w,2},c);
            solid(out,{r.x,r.y,2,r.h},c);
            solid(out,{r.x+r.w-2,r.y,2,r.h},c);
        }
        void text(draw_list& out,int x,int y,std::string content,
                  color c=foreground,
                  draw_command::alignment alignment=draw_command::alignment::left)
        {
            draw_command item{};
            item.kind=draw_command::type::text;
            item.text_alignment=alignment;
            item.x=x;item.y=y;item.tint=c;item.scale=text_scale;
            item.text=std::move(content);
            out.push_back(std::move(item));
        }
        std::vector<std::string> wrap(const std::string& value,int columns)
        {
            columns=std::max(1,columns);
            std::vector<std::string> lines;
            std::string line,word;
            const auto append=[&](const std::string& item)
            {
                std::size_t offset=0;
                while(offset<item.size())
                {
                    const auto remaining=std::size_t(columns)-line.size();
                    if(!remaining)
                    {
                        lines.push_back(line);
                        line.clear();
                        continue;
                    }
                    const auto count=std::min(remaining,item.size()-offset);
                    line+=item.substr(offset,count);
                    offset+=count;
                    if(offset<item.size())
                    {
                        lines.push_back(line);
                        line.clear();
                    }
                }
            };
            const auto flush_word=[&]()
            {
                if(word.empty())return;
                if(!line.empty() && line.size()+1+word.size()>
                    std::size_t(columns))
                {
                    lines.push_back(line);
                    line.clear();
                }
                if(!line.empty())line+=' ';
                append(word);
                word.clear();
            };
            for(char c:value)
            {
                if(c==' ' || c=='\n')
                {
                    flush_word();
                    if(c=='\n')
                    {
                        lines.push_back(line);
                        line.clear();
                    }
                }
                else word+=c;
            }
            flush_word();
            if(!line.empty() || lines.empty())lines.push_back(line);
            return lines;
        }

        // Single source of geometry for all menu modes (including idle).
        rectangle editor_panel(int width,int height)
        {
            const int x=side_margin,y=side_margin;
            return {x,y,std::max(0,std::min(panel_width_limit,width-x*2)),
                    std::max(0,height-y*2)};
        }

        std::string display(const property& p)
        {
            if(p.kind==property_kind::boolean)
                return p.value!=0 ? "ON" : "OFF";
            if(p.kind==property_kind::choice)
            {
                const int index=int(std::round(p.value));
                return index>=0 && index<int(p.choices.size())
                    ? p.choices[std::size_t(index)] : "NONE";
            }
            std::ostringstream out;
            if(p.kind==property_kind::integer)
                out<<std::llround(p.value);
            else
                out<<std::fixed<<std::setprecision(3)<<p.value;
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
        std::lock_guard<std::mutex> lock(mutex_);
        catalog_=std::move(types);
        selected_row_=std::clamp(selected_row_,0,
            std::max(0,int(catalog_.size())-1));
    }
    void session::set_labels(std::vector<screen_label> labels)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        labels_=std::move(labels);
    }
    void session::set_properties(std::uint64_t object,std::vector<property> values)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        properties_[object]=std::move(values);
        if(menu_==menu::properties && selected_object_==object)
            selected_row_=std::clamp(selected_row_,0,
                std::max(0,int(properties_[object].size())-1));
    }
    void session::set_frame_time(double milliseconds)
    {
        if(!std::isfinite(milliseconds) || milliseconds<=0)return;
        std::lock_guard<std::mutex> lock(mutex_);
        // Smoothing avoids rapidly changing numbers dominating the overlay.
        frame_time_ms_=frame_time_ms_>0
            ? frame_time_ms_*0.85+milliseconds*0.15 : milliseconds;
    }

    void session::handle(const input& events)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        // TAB works irrespective of editor mode and never changes selection.
        if(events.tab)cursor_free_=!cursor_free_;
        if(events.question)
        {
            visible_=!visible_;
            menu_=menu::none;
            selected_row_=0;
            scroll_first_=0;
        }
        if(!visible_)return;
        if(events.escape)
        {
            menu_=menu::none;
            return;
        }
        if(menu_==menu::insert)
        {
            if(!catalog_.empty())
            {
                if(events.up)selected_row_=(selected_row_+
                    int(catalog_.size())-1)%int(catalog_.size());
                if(events.down)selected_row_=
                    (selected_row_+1)%int(catalog_.size());
                if(events.enter)
                {
                    pending_.push_back({action_kind::insert,
                        catalog_[std::size_t(selected_row_)].id,0,"",0});
                    menu_=menu::none;
                }
            }
            return;
        }
        if(menu_==menu::properties)
        {
            auto found=properties_.find(selected_object_);
            if(found==properties_.end() || found->second.empty())return;
            auto& list=found->second;
            if(events.up)selected_row_=
                (selected_row_+int(list.size())-1)%int(list.size());
            if(events.down)selected_row_=
                (selected_row_+1)%int(list.size());
            if(events.left || events.right)
            {
                property& p=list[std::size_t(selected_row_)];
                const double old=p.value;
                if(p.kind==property_kind::boolean)
                    p.value=p.value==0?1:0;
                else
                {
                    const double step=p.kind==property_kind::choice ||
                        p.kind==property_kind::integer ? std::max(1.0,p.step) :
                        (p.step>0?p.step:0.1);
                    const double maximum=p.kind==property_kind::choice ?
                        double(std::max(0,int(p.choices.size())-1)) : p.maximum;
                    p.value=std::clamp(p.value+
                        (events.right?step:-step),std::min(p.minimum,maximum),
                        std::max(p.minimum,maximum));
                    if(p.kind==property_kind::integer ||
                       p.kind==property_kind::choice)
                        p.value=std::round(p.value);
                }
                if(old!=p.value)
                    pending_.push_back({action_kind::edit,"",
                        selected_object_,p.id,p.value});
            }
            return;
        }
        if(events.enter)
        {
            menu_=menu::insert;
            selected_row_=0;
            scroll_first_=0;
            return;
        }
        if(events.click && cursor_free_)
        {
            // Reverse order gives later (foreground) labels priority.
            for(auto it=labels_.rbegin();it!=labels_.rend();++it)
            {
                if(!it->visible || !it->object)continue;
                // Labels are screen-space; the host sends actual draw extent
                // with set_labels and should clip invisible 3D objects.
                const auto r=label_rect(*it,events.viewport_width,events.viewport_height);
                if(events.mouse_x>=r.x && events.mouse_x<r.x+r.w &&
                   events.mouse_y>=r.y && events.mouse_y<r.y+r.h)
                {
                    selected_object_=it->object;
                    selected_row_=0;
                    scroll_first_=0;
                    menu_=menu::properties;
                    break;
                }
            }
        }
    }

    std::vector<action> session::take_actions()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<action> result;
        result.swap(pending_);
        return result;
    }

    draw_list session::paint(int width,int height) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        draw_list out;
        if(!visible_ || width<=0 || height<=0)return out;

        for(const auto& item:labels_)
        {
            if(!item.visible || !item.object)continue;
            const auto r=label_rect(item,width,height);
            solid(out,r,selected_object_==item.object?
                selected_label_color:label_color);
            border(out,r,frame);
            text(out,r.x+10,r.y+8,item.text);
        }

        // A single full-height panel replaces the former temporary status
        // bar and short content-sized popups. No key legends are rendered.
        const rectangle panel=editor_panel(width,height);
        if(panel.w<80 || panel.h<100)return out;
        solid(out,panel,background);
        border(out,panel,frame);
        const char* heading=menu_==menu::insert?"----- INSERT -----":
            menu_==menu::properties?"----- SELECT -----":"----- EDITOR -----";
        text(out,panel.x+panel.w/2,panel.y+38,heading,foreground,
             draw_command::alignment::center);

        // Independent top-right counter, outside the scene's HDR pipeline.
        if(frame_time_ms_>0 && width>panel.x+panel.w+170)
        {
            std::ostringstream timing;
            timing<<std::fixed<<std::setprecision(1)
                  <<1000.0/frame_time_ms_<<" FPS";
            std::ostringstream frame;
            frame<<std::fixed<<std::setprecision(2)
                 <<frame_time_ms_<<" MS";
            const auto right=width-24;
            text(out,right,26,timing.str(),foreground,
                 draw_command::alignment::right);
            text(out,right,54,frame.str(),foreground,
                 draw_command::alignment::right);
        }

        if(menu_==menu::none)return out;
        const std::vector<property>* selected_properties=nullptr;
        if(menu_==menu::properties)
        {
            const auto it=properties_.find(selected_object_);
            if(it!=properties_.end())selected_properties=&it->second;
        }
        const int rows=menu_==menu::insert?int(catalog_.size()):
            selected_properties?int(selected_properties->size()):0;
        const int content_x=panel.x+panel_padding;
        const int content_width=std::max(12,panel.w-2*panel_padding);
        const int columns=std::max(1,content_width/12);
        const int content_y=panel.y+header_height;
        const int content_bottom=panel.y+panel.h-panel_padding;
        if(content_bottom<=content_y)return out;
        if(rows==0)
        {
            text(out,content_x,content_y,
                 menu_==menu::insert?"NO TYPES REGISTERED":"NO PROPERTIES");
            return out;
        }

        std::vector<std::vector<std::string>> wrapped;
        std::vector<int> heights;
        wrapped.reserve(std::size_t(rows));
        heights.reserve(std::size_t(rows));
        for(int i=0;i<rows;++i)
        {
            std::string value;
            if(menu_==menu::insert)
            {
                const auto& item=catalog_[std::size_t(i)];
                value=item.name;
                if(!item.group.empty())value+=" ["+item.group+"]";
            }
            else
            {
                const auto& item=(*selected_properties)[std::size_t(i)];
                value=item.name+": "+display(item);
            }
            wrapped.push_back(wrap(value,columns));
            heights.push_back(int(wrapped.back().size())*line_height+14);
        }

        const int selected=std::clamp(selected_row_,0,rows-1);
        int first=std::clamp(scroll_first_,0,rows-1);
        if(selected<first)first=selected;
        const auto last_visible=[&](int origin)
        {
            int y=content_y;
            for(int i=origin;i<rows;++i)
            {
                if(y+heights[std::size_t(i)]>content_bottom)return i;
                y+=heights[std::size_t(i)];
            }
            return rows;
        };
        while(first<selected && selected>=last_visible(first))++first;
        scroll_first_=first;
        int y=content_y;
        for(int i=first;i<rows && y<content_bottom;++i)
        {
            const int block=heights[std::size_t(i)];
            if(i==selected)
                solid(out,{content_x-12,y-5,content_width+24,
                            std::min(block,content_bottom-y+5)},highlight);
            for(std::size_t j=0;j<wrapped[std::size_t(i)].size();++j)
            {
                const int line_y=y+int(j)*line_height;
                if(line_y+14>content_bottom)break;
                text(out,content_x,line_y,wrapped[std::size_t(i)][j]);
            }
            y+=block;
        }
        return out;
    }
    bool session::visible() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return visible_;
    }
    bool session::cursor_free() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return cursor_free_;
    }
    bool session::captures_input() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return cursor_free_ || (visible_ && menu_!=menu::none);
    }
}
