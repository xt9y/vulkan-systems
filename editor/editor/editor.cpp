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
        constexpr int row_height=27;
        constexpr int text_scale=2;

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
        void text(draw_list& out,int x,int y,std::string content,color c=foreground)
        {
            draw_command item{};
            item.kind=draw_command::type::text;
            item.x=x;item.y=y;item.tint=c;item.scale=text_scale;
            item.text=std::move(content);
            out.push_back(std::move(item));
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
        if(menu_==menu::none)
        {
            // Empty editor instances still show that '?' successfully
            // entered debug mode; scene labels are exclusively host-defined.
            if(labels_.empty())
            {
                const int w=std::min(345,width-12);
                solid(out,{6,6,w,36},background);
                border(out,{6,6,w,36},frame);
                text(out,17,17,"EDITOR  ENTER:INSERT  TAB:CURSOR");
            }
            return out;
        }
        const int left=22,top=22;
        const int panel_width=std::max(100,std::min(470,width-2*left));
        const int rows=menu_==menu::insert?int(catalog_.size()):
            (properties_.count(selected_object_)
                ?int(properties_.at(selected_object_).size()):0);
        const int visible_rows=std::min(rows,
            std::max(0,(height-top-102)/row_height));
        const int panel_height=std::min(height-top-8,
            93+std::max(1,visible_rows)*row_height);
        if(panel_height<=0)return out;
        solid(out,{left,top,panel_width,panel_height},background);
        border(out,{left,top,panel_width,panel_height},frame);
        text(out,left+18,top+20,menu_==menu::insert?
            "----- INSERT -----":"----- PROPERTIES -----");
        const int start=visible_rows>0?
            std::clamp(selected_row_-visible_rows+1,0,
                std::max(0,rows-visible_rows)):0;
        for(int row=start;row<start+visible_rows;++row)
        {
            const int y=top+64+(row-start)*row_height;
            if(row==selected_row_)
                solid(out,{left+8,y-4,panel_width-16,row_height},highlight);
            std::string line=row==selected_row_?"> ":"  ";
            if(menu_==menu::insert)
            {
                const auto& item=catalog_[std::size_t(row)];
                line+=item.name;
                if(!item.group.empty())line+=" ["+item.group+"]";
            }
            else
            {
                const auto& value=properties_.at(selected_object_)[std::size_t(row)];
                line+=value.name+": "+display(value);
            }
            const int max_chars=std::max(1,(panel_width-36)/12);
            if(int(line.size())>max_chars)line.resize(std::size_t(max_chars));
            text(out,left+16,y,line);
        }
        if(rows==0)
            text(out,left+16,top+64,menu_==menu::insert?
                "NO TYPES REGISTERED":"NO PROPERTIES");
        text(out,left+16,top+panel_height-24,
            menu_==menu::insert?"ARROWS / ENTER / ESC":"ARROWS / LEFT RIGHT / ESC");
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
