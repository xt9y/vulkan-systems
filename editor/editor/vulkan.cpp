#include "vulkan.hpp"
#include "uploaded_font.hpp"
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
            std::string ui_directory="build/editor-hell2024/"
                "Hell2024/Hell2024/res/textures/ui";
            std::array<glyph,128> glyphs;
            std::array<sprite,10> sprites;
            bool ready=false;
        };
        font_data& font()
        {
            static font_data data;
            return data;
        }
        // Times New Roman Bold coverage matching the original light sign.
        constexpr std::uint8_t cube_letters[10][34]={
            {0,0,0,0,16,32,0,0,0,0,0,0,16,32,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
            {0,0,112,208,240,224,224,128,0,0,16,192,240,224,224,112,0,32,192,240,240,240,240,208,80,32,192,240,240,224,240,240,240,0},
            {0,128,240,224,32,0,16,176,80,0,160,240,176,0,16,208,64,0,48,240,240,16,96,240,240,48,48,240,240,32,0,64,240,32},
            {0,240,240,112,0,0,0,0,0,0,208,240,112,0,0,112,112,0,32,240,240,0,32,240,240,96,32,240,240,32,176,0,112,48},
            {64,240,240,48,0,0,0,0,0,0,224,240,96,0,0,80,128,0,32,240,240,64,160,240,224,0,32,240,240,128,240,0,0,0},
            {64,240,240,48,0,0,0,0,0,0,224,240,96,0,0,80,128,0,32,240,240,224,240,240,80,0,32,240,240,224,240,0,0,0},
            {0,240,240,96,0,0,0,32,80,0,224,240,96,0,0,80,128,0,32,240,240,0,144,240,224,0,32,240,240,32,208,0,64,0},
            {0,144,240,192,0,0,0,176,128,0,224,240,96,0,0,80,128,0,32,240,240,0,96,240,240,0,32,240,240,32,64,48,176,0},
            {0,0,144,240,208,128,192,240,128,64,240,240,176,32,32,176,208,32,128,240,240,160,240,240,128,0,128,240,240,144,144,240,176,0},
            {0,0,0,16,96,112,64,32,48,64,80,80,80,32,32,80,80,48,80,80,80,80,64,32,0,16,80,80,80,80,80,80,48,0}
        };

        sprite load_sprite(const std::string& filename,int w,int h,
                           bool cube_plate=false,bool empty_plate=false)
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
                // Reuse the light sign's blank edge texture for other labels,
                // retaining the original end caps and the textured row colors.
                const int source_x=(cube_plate || empty_plate) && x>=8 && x<w-8 ?
                    2+(x-8)%6 : x;
                const int px=std::min(sw-1,source_x*sw/w);
                const int py=std::min(sh-1,y*sh/h);
                const std::size_t i=4u*(std::size_t(py)*std::size_t(sw)+
                    std::size_t(px));
                const unsigned alpha=pixels[i+3];
                if(alpha<64)return std::uint32_t(0);
                // Quantize to original 16-color-channel steps to reduce
                // Vulkan clear attachment color changes.
                const int ink_x=x*68/w-17,ink_y=y*33/h-11;
                const unsigned ink=cube_plate && ink_x>=0 && ink_x<34 && ink_y>=0 && ink_y<10
                    ?cube_letters[ink_y][ink_x]:0u;
                const auto channel=[&](int k)
                {
                    const unsigned value=(unsigned(pixels[i+k])*(255u-ink)+224u*ink)/255u;
                    return (value*alpha/255u)&0xf0u;
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

            // The uploaded x_font.png is a 16x16 CP437 grid, with each
            // character occupying 9x16 pixels. Its 1-bit pixel mask is
            // embedded in uploaded_font.hpp; decode once at startup.
            const std::string encoded=uploaded_font::encoded_png;
            std::vector<stbi_uc> png;
            png.reserve(encoded.size()*3/4);
            unsigned int bits=0;
            int count=0;
            for(unsigned char c:encoded)
            {
                if(c=='=')break;
                int value=-1;
                if(c>='A'&&c<='Z')value=c-'A';
                else if(c>='a'&&c<='z')value=c-'a'+26;
                else if(c>='0'&&c<='9')value=c-'0'+52;
                else if(c=='+')value=62;
                else if(c=='/')value=63;
                if(value<0)
                    throw std::runtime_error("Corrupted embedded x_font base64");
                bits=(bits<<6)|static_cast<unsigned int>(value);
                count+=6;
                if(count>=8)
                {
                    count-=8;
                    png.push_back(static_cast<stbi_uc>((bits>>count)&255u));
                }
            }
            int width=0,height=0,channels=0;
            stbi_uc* pixels=stbi_load_from_memory(png.data(),
                static_cast<int>(png.size()),&width,&height,&channels,4);
            if(!pixels||width!=144||height!=256)
            {
                stbi_image_free(pixels);
                throw std::runtime_error("Embedded x_font must be 144x256");
            }
            std::array<glyph,128> loaded{};
            for(unsigned int character=33;character<=126;++character)
            {
                const int x0=int(character%16u)*9;
                const int y0=int(character/16u)*16;
                glyph entry;
                entry.width=8; // 7-pixel ink width and one spacing pixel.
                entry.height=16;
                entry.present=true;
                for(int y=0;y<16;++y)
                {
                    for(int x=0;x<9;)
                    {
                        const auto ink=[&](int column)
                        {
                            const std::size_t offset=4u*(
                                std::size_t(y0+y)*144u+
                                std::size_t(x0+column));
                            return pixels[offset]>127;
                        };
                        if(!ink(x)){++x;continue;}
                        const int first=x;
                        do{++x;}while(x<9&&ink(x));
                        entry.spans.push_back({
                            static_cast<std::uint16_t>(first),
                            static_cast<std::uint16_t>(y),
                            static_cast<std::uint16_t>(x-first)});
                    }
                }
                loaded[character]=std::move(entry);
            }
            stbi_image_free(pixels);
            const std::array<const char*,10> names{{
                "MenuBG.png","MenuBorderHorizontal.png",
                "MenuBorderVertical.png","MenuBorderCornerTL.png",
                "MenuBorderCornerTR.png","MenuBorderCornerBL.png",
                "MenuBorderCornerBR.png","Icon_Light.jpg","Icon_Light.jpg",
                "Icon_Light.jpg"
            }};
            const std::array<std::array<int,2>,10> sizes{{
                {{200,245}},{{200,3}},{{3,245}},
                {{4,4}},{{4,4}},{{4,4}},{{4,4}},{{62,22}},{{62,22}},{{62,22}}
            }};
            std::array<sprite,10> ui{};
            for(std::size_t i=0;i<names.size();++i)
                ui[i]=load_sprite(atlas.ui_directory+"/"+names[i],
                    sizes[i][0],sizes[i][1],
                    i==static_cast<std::size_t>(draw_command::texture::cube_icon),
                    i==static_cast<std::size_t>(draw_command::texture::label_plate));

            atlas.glyphs=std::move(loaded);
            atlas.sprites=std::move(ui);
            atlas.ready=true;
            std::clog<<"editor: preloaded uploaded x_font.png CP437 atlas "
                     <<"and Hell2024 UI textures\\n";
        }

        bool same(color a,color b)
        {
            return a.r==b.r && a.g==b.g && a.b==b.b;
        }
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
        // The original editor renders at 960x540 and scales its entire UI
        // to the presentation framebuffer. The old one-pixel implementation
        // bypassed this transformation and made all text microscopic.
        const double scale_x=double(extent.width)/960.0;
        const double scale_y=double(extent.height)/540.0;
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
        const auto rectangle=[&](double x,double y,double w,double h,color tint)
        {
            if(w<=0 || h<=0)return;
            const auto clip=[&](double v,int limit)
            {
                return static_cast<int>(std::clamp(std::lround(v),
                    0L,static_cast<long>(limit)));
            };
            const int x0=clip(x*scale_x,int(extent.width));
            const int y0=clip(y*scale_y,int(extent.height));
            const int x1=clip((x+w)*scale_x,int(extent.width));
            const int y1=clip((y+h)*scale_y,int(extent.height));
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
            if(item.kind==draw_command::type::image)
            {
                const auto index=static_cast<std::size_t>(item.image);
                if(index>=atlas.sprites.size())continue;
                const auto& sprite=atlas.sprites[index];
                if(sprite.width<=0 || sprite.height<=0)continue;
                const double image_x=double(item.width)/sprite.width;
                const double image_y=double(item.height)/sprite.height;
                for(const auto& span:sprite.runs)
                {
                    const auto rgb=span.rgb;
                    color tint{
                        float((rgb>>16)&255u)/255.0f,
                        float((rgb>>8)&255u)/255.0f,
                        float(rgb&255u)/255.0f,1.0f};
                    if(item.image==draw_command::texture::menu_bg)
                    {
                        // Keep the original sprite's texture pattern,
                        // but shift its saturated blue palette toward
                        // Hell's more neutral, dark purple-grey.
                        const float luma=0.2126f*tint.r+
                            0.7152f*tint.g+0.0722f*tint.b;
                        tint={0.078f+0.22f*luma,
                              0.069f+0.22f*luma,
                              0.104f+0.24f*luma,1.0f};
                    }
                    else if(item.image!=draw_command::texture::light_icon &&
                            item.image!=draw_command::texture::cube_icon &&
                            item.image!=draw_command::texture::label_plate)
                    {
                        // Borders and corners should be grey rather than
                        // bright white, without modifying their geometry.
                        tint.r*=0.80f;
                        tint.g*=0.80f;
                        tint.b*=0.82f;
                    }
                    // Apply hover/selection tint to sign backgrounds while
                    // keeping their baked light lettering readable.
                    const bool sign=item.image==draw_command::texture::light_icon ||
                                    item.image==draw_command::texture::cube_icon ||
                                    item.image==draw_command::texture::label_plate;
                    const bool lettering=sign && std::min({tint.r,tint.g,tint.b})>0.6f;
                    if(!lettering)
                    {
                        tint.r*=item.tint.r;
                        tint.g*=item.tint.g;
                        tint.b*=item.tint.b;
                    }
                    rectangle(item.x+span.x*image_x,
                              item.y+span.y*image_y,
                              span.length*image_x,image_y,tint);
                }
                continue;
            }
            const int scale=std::clamp(item.scale,1,6);
            int advance=0;
            for(unsigned char letter:item.text)
            {
                if(letter=='\n')break;
                if(letter==' '){advance+=6*scale;continue;}
                const auto& entry=atlas.glyphs[letter<128?letter:0];
                if(entry.present)advance+=entry.width*scale;
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
                    pen_y+=16*scale;
                    continue;
                }
                if(letter==' ')
                {
                    pen_x+=6*scale;
                    continue;
                }
                // Only original font glyphs; no geometric fallback.
                const auto& entry=atlas.glyphs[letter<128?letter:0];
                if(!entry.present)continue;
                for(const auto& span:entry.spans)
                    rectangle(pen_x+span.x*scale,pen_y+span.y*scale,
                              span.length*scale,scale,item.tint);
                pen_x+=entry.width*scale;
            }
        }
        flush();
    }
}
