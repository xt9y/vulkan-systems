#include "vulkan.hpp"
#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <vector>
#include <string>
#include <iostream>
#include <mutex>
#include <utility>
#define STBI_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

namespace editor::vulkan
{
    namespace
    {
        using glyph_rows=std::array<std::uint8_t,7>;
        // Built-in compact 5x7 block alphabet. Deliberately independent of
        // textures, fonts, descriptor sets, third-party UI or shader programs.
        glyph_rows glyph(unsigned char c)
        {
            switch(std::toupper(c))
            {
            case 'A':return {14,17,17,31,17,17,17};
            case 'B':return {30,17,17,30,17,17,30};
            case 'C':return {14,17,16,16,16,17,14};
            case 'D':return {30,17,17,17,17,17,30};
            case 'E':return {31,16,16,30,16,16,31};
            case 'F':return {31,16,16,30,16,16,16};
            case 'G':return {15,16,16,23,17,17,15};
            case 'H':return {17,17,17,31,17,17,17};
            case 'I':return {31,4,4,4,4,4,31};
            case 'J':return {7,2,2,2,18,18,12};
            case 'K':return {17,18,20,24,20,18,17};
            case 'L':return {16,16,16,16,16,16,31};
            case 'M':return {17,27,21,21,17,17,17};
            case 'N':return {17,25,21,19,17,17,17};
            case 'O':return {14,17,17,17,17,17,14};
            case 'P':return {30,17,17,30,16,16,16};
            case 'Q':return {14,17,17,17,21,18,13};
            case 'R':return {30,17,17,30,20,18,17};
            case 'S':return {15,16,16,14,1,1,30};
            case 'T':return {31,4,4,4,4,4,4};
            case 'U':return {17,17,17,17,17,17,14};
            case 'V':return {17,17,17,17,17,10,4};
            case 'W':return {17,17,17,21,21,21,10};
            case 'X':return {17,17,10,4,10,17,17};
            case 'Y':return {17,17,10,4,4,4,4};
            case 'Z':return {31,1,2,4,8,16,31};
            case '0':return {14,17,19,21,25,17,14};
            case '1':return {4,12,4,4,4,4,14};
            case '2':return {14,17,1,2,4,8,31};
            case '3':return {30,1,1,14,1,1,30};
            case '4':return {2,6,10,18,31,2,2};
            case '5':return {31,16,16,30,1,1,30};
            case '6':return {14,16,16,30,17,17,14};
            case '7':return {31,1,2,4,8,8,8};
            case '8':return {14,17,17,14,17,17,14};
            case '9':return {14,17,17,15,1,1,14};
            case '?':return {14,17,1,2,4,0,4};
            case '!':return {4,4,4,4,4,0,4};
            case '.':return {0,0,0,0,0,6,6};
            case ',':return {0,0,0,0,6,6,4};
            case ':':return {0,4,4,0,4,4,0};
            case ';':return {0,4,4,0,4,4,8};
            case '-':return {0,0,0,31,0,0,0};
            case '_':return {0,0,0,0,0,0,31};
            case '+':return {0,4,4,31,4,4,0};
            case '/':return {1,2,2,4,8,8,16};
            case '\\':return {16,8,8,4,2,2,1};
            case '[':return {14,8,8,8,8,8,14};
            case ']':return {14,2,2,2,2,2,14};
            case '(':return {2,4,8,8,8,4,2};
            case ')':return {8,4,2,2,2,4,8};
            case '<':return {2,4,8,16,8,4,2};
            case '>':return {8,4,2,1,2,4,8};
            case '=':return {0,31,0,31,0,0,0};
            case '#':return {10,10,31,10,31,10,10};
            case '%':return {17,18,4,8,19,17,0};
            case '*':return {0,21,14,31,14,21,0};
            case ' ':return {};
            default:return {31,17,21,21,21,17,31};
            }
        }

        struct bitmap
        {
            int width=0,height=0;
            std::vector<std::uint8_t> pixels;
            bool valid() const{return width>0 && height>0 &&
                pixels.size()==std::size_t(width)*std::size_t(height);}
        };

        struct font_data
        {
            std::once_flag once;
            std::string directory="build/editor-font-repository/"
                "Rogue/Rogue/res/textures/font";
            std::array<bitmap,128> glyphs;
            bool available=false;
        };

        font_data& font()
        {
            static font_data instance;
            return instance;
        }

        // Exact index mapping from Rogue/Rogue/src/Core/TextBlitting.cpp.
        // Its UTF-8 placeholder bullets occupy three bytes each, so use
        // byte offsets (std::string::find) as the old text blitter did.
        void load_font(font_data& out)
        {
            unsigned int glyph_count=0;
            const std::string sheet=
                "•!\"#$%&'••*+,-./0123456789:;<=>?_"
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ\\^_`abcdefghijklmnopqrstuvwxyz";
            for(int character=33;character<127;++character)
            {
                const auto index=sheet.find(char(character));
                if(index==std::string::npos || index>=90)continue;
                const std::string filename=out.directory+"/char_"+
                    std::to_string(index+1)+".png";
                int w=0,h=0,channels=0;
                stbi_uc* pixels=stbi_load(filename.c_str(),&w,&h,
                                          &channels,4);
                if(!pixels || w<=0 || h<=0 || w>128 || h>128)
                {
                    stbi_image_free(pixels);
                    continue;
                }
                bitmap glyph;
                glyph.width=w;
                glyph.height=h;
                glyph.pixels.resize(std::size_t(w)*std::size_t(h));
                for(int i=0;i<w*h;++i)
                {
                    const int r=pixels[i*4+0],g=pixels[i*4+1];
                    const int b=pixels[i*4+2],a=pixels[i*4+3];
                    // The source contains individual white, translucent
                    // glyph PNGs. Use alpha and brightness as coverage.
                    glyph.pixels[std::size_t(i)]=std::uint8_t(
                        std::max({r,g,b})*a/255);
                }
                stbi_image_free(pixels);
                out.glyphs[std::size_t(character)]=std::move(glyph);
                out.available=true;
                ++glyph_count;
            }
            if(glyph_count)
                std::clog<<"editor: original HellEngine bitmap font: "
                         <<glyph_count<<" glyphs loaded\n";
        }

        bool same(color a,color b)
        {
            return a.r==b.r && a.g==b.g && a.b==b.b;
        }
    }

    void set_font_directory(std::string directory)
    {
        // This is an initialization-time configuration API. The underlying
        // font cache is populated once on the first rendering call.
        font().directory=std::move(directory);
    }

    void record(vk::CommandBuffer command,const draw_list& commands,
                vk::Extent2D extent,std::uint32_t color_attachment)
    {
        // Hidden editor: zero Vulkan commands and zero GPU/CPU allocations.
        if(commands.empty() || !command || !extent.width || !extent.height)return;
        auto& atlas=font();
        std::call_once(atlas.once,[&]{load_font(atlas);});
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
            const int scale=std::clamp(item.scale,1,5);
            int advance=0;
            for(unsigned char letter:item.text)
            {
                if(letter=='\n')break;
                if(letter==' ')
                {
                    advance+=atlas.available?6:6*scale;
                    continue;
                }
                const auto& g=letter<atlas.glyphs.size()?
                    atlas.glyphs[letter]:atlas.glyphs[0];
                advance+=g.valid()?g.width:6*scale;
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
                    pen_y+=atlas.available?24:8*scale;
                    continue;
                }
                if(letter==' ')
                {
                    pen_x+=atlas.available?6:6*scale;
                    continue;
                }
                const auto& bitmap_font=letter<atlas.glyphs.size()?
                    atlas.glyphs[letter]:atlas.glyphs[0];
                if(bitmap_font.valid())
                {
                    for(int row=0;row<bitmap_font.height;++row)
                    {
                        for(int column=0;column<bitmap_font.width;)
                        {
                            const auto inside=[&](int col)
                            {
                                return bitmap_font.pixels[
                                    std::size_t(row*bitmap_font.width+col)]>72;
                            };
                            if(!inside(column))
                            {
                                ++column;
                                continue;
                            }
                            const int start=column;
                            do{++column;}while(column<bitmap_font.width &&
                                              inside(column));
                            rectangle(pen_x+start,pen_y+row,
                                column-start,1,item.tint);
                        }
                    }
                    pen_x+=bitmap_font.width;
                    continue;
                }
                // Remain usable when the legacy PNGs have not been fetched.
                const auto bitmap=glyph(letter);
                for(int row=0;row<7;++row)
                {
                    for(int column=0;column<5;)
                    {
                        if(!(bitmap[std::size_t(row)]&(1u<<(4-column))))
                        {
                            ++column;
                            continue;
                        }
                        const int start=column;
                        do{++column;}while(column<5 &&
                            (bitmap[std::size_t(row)]&(1u<<(4-column))));
                        rectangle(pen_x+start*scale,pen_y+row*scale,
                            (column-start)*scale,scale,item.tint);
                    }
                }
                pen_x+=6*scale;
            }
        }
        flush();
    }
}
