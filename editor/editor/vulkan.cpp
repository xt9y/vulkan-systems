#include "vulkan.hpp"
#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <vector>

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

        bool same(color a,color b)
        {
            return a.r==b.r && a.g==b.g && a.b==b.b;
        }
    }

    void record(vk::CommandBuffer command,const draw_list& commands,
                vk::Extent2D extent,std::uint32_t color_attachment)
    {
        if(!command || !extent.width || !extent.height)return;
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
            int pen_x=item.x;
            int pen_y=item.y;
            for(unsigned char letter:item.text)
            {
                if(letter=='\n')
                {
                    pen_x=item.x;
                    pen_y+=8*scale;
                    continue;
                }
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
