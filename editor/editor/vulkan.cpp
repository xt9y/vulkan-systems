#include "vulkan.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>
#include <string>
#include <iostream>
#include <utility>
#include <stdexcept>
// stb_image uses STB_IMAGE_STATIC (without the extra I) to make
// all implementation symbols TU-local. VulkanGame has its own stb_image.
#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

namespace editor::vulkan
{
    namespace
    {
        struct glyph_span
        {
            std::uint16_t x=0,y=0,length=0;
        };
        struct glyph
        {
            int width=0,height=0;
            std::vector<glyph_span> spans;
            bool present=false;
        };
        struct colored_span
        {
            std::uint16_t x=0,y=0,length=0;
            std::uint32_t rgb=0;
        };
        struct sprite
        {
            int width=0,height=0;
            std::vector<colored_span> runs;
        };
        struct font_data
        {
            std::string directory="build/editor-hell2024/"
                "Hell2024/Hell2024/res/textures/font";
            std::string ui_directory="build/editor-hell2024/"
                "Hell2024/Hell2024/res/textures/ui";
            std::array<glyph,128> glyphs;
            std::array<sprite,8> sprites;
            bool ready=false;
        };
        font_data& font()
        {
            static font_data data;
            return data;
        }
        // Exact byte-oriented mapping from Hell2024/src/Renderer/TextBlitting.cpp.
        const std::string& font_sheet()
        {
            static const std::string sheet=
                "•!\"#$%&'••*+,-./0123456789:;<=>?_"
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ\\^_`abcdefghijklmnopqrstuvwxyz";

            return sheet;
        }
        sprite load_sprite(const std::string& filename,int w,int h)
        {
            int sw=0,sh=0,channels=0;
            stbi_uc* pixels=stbi_load(filename.c_str(),&sw,&sh,&channels,4);
            if(!pixels||sw<=0||sh<=0||sw>4096||sh>4096)
            {
                stbi_image_free(pixels);
                throw std::runtime_error(
                    "VulkanEditor missing original Hell2024 UI texture: "+filename);
            }
            sprite result;
            result.width=w;
            result.height=h;
            const auto sample=[&](int x,int y)
            {
                const int px=std::min(sw-1,x*sw/w);
                const int py=std::min(sh-1,y*sh/h);
                const std::size_t i=4u*(std::size_t(py)*std::size_t(sw)+
                    std::size_t(px));
                const unsigned alpha=pixels[i+3];
                if(alpha<64)return std::uint32_t(0);
                // Quantize to original 16-color-channel steps to reduce
                // Vulkan clear attachment color changes.
                const auto channel=[&](int k)
                {
                    return (unsigned(pixels[i+k])*alpha/255u)&0xf0u;
                };
                return std::uint32_t(0xff000000u|
                    (channel(0)<<16)|(channel(1)<<8)|channel(2));
            };
            for(int y=0;y<h;++y)
                for(int x=0;x<w;)
                {
                    const auto rgb=sample(x,y);
                    if(!rgb){++x;continue;}
                    const int begin=x;
                    do{++x;}while(x<w&&sample(x,y)==rgb);
                    result.runs.push_back({
                        static_cast<std::uint16_t>(begin),
                        static_cast<std::uint16_t>(y),
                        static_cast<std::uint16_t>(x-begin),rgb});
                }
            stbi_image_free(pixels);
            // Sprite texels do not overlap: color sorting preserves output.
            std::stable_sort(result.runs.begin(),result.runs.end(),
                [](const colored_span& a,const colored_span& b)
                {return a.rgb<b.rgb;});
            return result;
        }

        void load_font(font_data& atlas)
        {
            std::array<glyph,128> loaded{};
            const std::string& sheet=font_sheet();
            // Decode the complete original 90-image font, before drawing.
            for(std::size_t i=0;i<90;++i)
            {
                const std::string filename=atlas.directory+"/char_"+
                    std::to_string(i+1)+".png";
                int width=0,height=0,channels=0;
                stbi_uc* pixels=stbi_load(filename.c_str(),&width,&height,
                                          &channels,4);
                if(!pixels || width<=0 || height<=0 ||
                   width>128 || height>128)
                {
                    stbi_image_free(pixels);
                    throw std::runtime_error(
                        "VulkanEditor original font missing or invalid: "+filename+
                        " (fetch original Hell2024 editor assets before running)");
                }
                glyph entry;
                entry.width=width;
                entry.height=height;
                entry.present=true;
                for(int y=0;y<height;++y)
                {
                    for(int x=0;x<width;)
                    {
                        const auto coverage=[&](int col)
                        {
                            const int index=(y*width+col)*4;
                            const int r=pixels[index],g=pixels[index+1];
                            const int b=pixels[index+2],a=pixels[index+3];
                            return std::max({r,g,b})*a/255>72;
                        };
                        if(!coverage(x)){++x;continue;}
                        const int begin=x;
                        do{++x;}while(x<width && coverage(x));
                        entry.spans.push_back({
                            static_cast<std::uint16_t>(begin),
                            static_cast<std::uint16_t>(y),
                            static_cast<std::uint16_t>(x-begin)});
                    }
                }
                stbi_image_free(pixels);
                // The original character map contains three-byte UTF-8
                // placeholders; only printable ASCII slots are used here.
                if(i<sheet.size())
                {
                    const unsigned char code=
                        static_cast<unsigned char>(sheet[i]);
                    if(code>=33 && code<127 &&
                       sheet.find(char(code))==i)
                        loaded[code]=std::move(entry);
                }
            }
            // The backup contains char_1..char_90, while the original
            // mapping refers to five additional lowercase glyphs (v-z).
            // Reuse their authentic uppercase PNGs, never a synthetic font.
            for(unsigned char c='v';c<='z';++c)
                loaded[c]=loaded[static_cast<unsigned char>(c-'a'+'A')];
            const std::array<const char*,8> names{{
                "MenuBG.png","MenuBorderHorizontal.png",
                "MenuBorderVertical.png","MenuBorderCornerTL.png",
                "MenuBorderCornerTR.png","MenuBorderCornerBL.png",
                "MenuBorderCornerBR.png","Icon_Light.jpg"
            }};
            const std::array<std::array<int,2>,8> sizes{{
                {{200,245}},{{200,3}},{{3,245}},
                {{4,4}},{{4,4}},{{4,4}},{{4,4}},{{62,22}}
            }};
            std::array<sprite,8> ui{};
            for(std::size_t i=0;i<names.size();++i)
                ui[i]=load_sprite(atlas.ui_directory+"/"+names[i],
                    sizes[i][0],sizes[i][1]);

            atlas.glyphs=std::move(loaded);
            atlas.sprites=std::move(ui);
            atlas.ready=true;
            std::clog<<"editor: preloaded Hell2024 original bitmap font "
                     <<"and 8 original UI sprites\n";
        }

        bool same(color a,color b)
        {
            return a.r==b.r && a.g==b.g && a.b==b.b;
        }
    }

    void set_font_directory(std::string directory)
    {
        if(font().ready)
            throw std::logic_error(
                "Editor font directory cannot change after initialization");
        font().directory=std::move(directory);
    }

    void set_ui_directory(std::string directory)
    {
        if(font().ready)
            throw std::logic_error(
                "Editor UI directory cannot change after initialization");
        font().ui_directory=std::move(directory);
    }

    std::array<int,128> font_metrics()
    {
        if(!font().ready)
            throw std::logic_error("Editor font must be initialized first");
        std::array<int,128> widths{};
        widths[' ']=6;
        for(std::size_t i=0;i<widths.size();++i)
            if(font().glyphs[i].present)
                widths[i]=font().glyphs[i].width;
        return widths;
    }

    void initialize_font()
    {
        if(!font().ready)load_font(font());
    }

    void record(vk::CommandBuffer command,const draw_list& commands,
                vk::Extent2D extent,std::uint32_t color_attachment)
    {
        if(commands.empty() || !command || !extent.width || !extent.height)return;
        const auto& atlas=font();
        if(!atlas.ready)
            throw std::logic_error(
                "VulkanEditor::initialize_font must run before rendering");
        std::vector<vk::ClearRect> batch;
        batch.reserve(1024);
        color current{};
        bool has_color=false;

        const auto flush=[&]()
        {
            if(batch.empty())return;
            vk::ClearAttachment attachment{};
            attachment.aspectMask=vk::ImageAspectFlagBits::eColor;
            attachment.colorAttachment=color_attachment;
            attachment.clearValue.color=vk::ClearColorValue(
                std::array<float,4>{current.r,current.g,current.b,1.f});
            command.clearAttachments(
                1,&attachment,static_cast<std::uint32_t>(batch.size()),batch.data());
            batch.clear();
        };
        const auto rectangle=[&](int x,int y,int w,int h,color tint)
        {
            if(w<=0 || h<=0)return;
            const std::int64_t right=std::int64_t(x)+w;
            const std::int64_t bottom=std::int64_t(y)+h;
            const int x0=std::max(0,x),y0=std::max(0,y);
            const int x1=static_cast<int>(std::min<std::int64_t>(
                extent.width,right));
            const int y1=static_cast<int>(std::min<std::int64_t>(
                extent.height,bottom));
            if(x1<=x0 || y1<=y0)return;
            if(!has_color || !same(current,tint))
            {
                flush();
                current=tint;
                has_color=true;
            }
            vk::ClearRect rect{};
            rect.rect=vk::Rect2D{{x0,y0},
                {static_cast<std::uint32_t>(x1-x0),
                 static_cast<std::uint32_t>(y1-y0)}};
            rect.baseArrayLayer=0;
            rect.layerCount=1;
            batch.push_back(rect);
            if(batch.size()>=2048)flush();
        };

        for(const auto& item:commands)
        {
            if(item.kind==draw_command::type::rectangle)
            {
                rectangle(item.x,item.y,item.width,item.height,item.tint);
                continue;
            }
            int advance=0;
            for(unsigned char letter:item.text)
            {
                if(letter=='\n')break;
                if(letter==' '){advance+=6;continue;}
                const auto& entry=atlas.glyphs[letter<128?letter:0];
                if(entry.present)advance+=entry.width;
            }
            int start_x=item.x;
            if(item.text_alignment==draw_command::alignment::center)
                start_x-=advance/2;
            else if(item.text_alignment==draw_command::alignment::right)
                start_x-=advance;
            int pen_x=start_x;
            int pen_y=item.y;
            for(unsigned char letter:item.text)
            {
                if(letter=='\n')
                {
                    pen_x=start_x;
                    pen_y+=24;
                    continue;
                }
                if(letter==' ')
                {
                    pen_x+=6;
                    continue;
                }
                // Only original font glyphs; no geometric fallback.
                const auto& entry=atlas.glyphs[letter<128?letter:0];
                if(!entry.present)continue;
                for(const auto& span:entry.spans)
                    rectangle(pen_x+span.x,pen_y+span.y,
                              span.length,1,item.tint);
                pen_x+=entry.width;
            }
        }
        flush();
    }
}
