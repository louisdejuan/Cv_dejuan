#ifndef _SDL2_COLOR_HPP_
#define _SDL2_COLOR_HPP_
#include <cstdint>
#include <array>
#include <ostream>

namespace sdl2
{
    struct color
    {
        color()
            : red(0x00), green(0x00), blue(0x00), alpha(0xFF)
        {}
        color(int r, int g, int b, int a = 0xFF)
            : red(r), green(g), blue(b), alpha(a)
        {}

        color operator+(const color c)
        {
            return {255*(red + c.red >= 255) + (red + c.red)*(red + c.red < 255), 255*(green + c.green >= 255) + (green + c.green)*(green + c.green < 255), 255*(blue + c.blue >= 255) + (blue + c.blue)*(blue + c.blue < 255), (int)((alpha + c.alpha)/2)};
        }

        color operator-(const color c)
        {
            return {(red - c.red)*(red - c.red > 0), (green - c.green)*(green - c.green > 0), (blue - c.blue)*(blue - c.blue > 0), (int)((alpha + c.alpha)/2)};
        }

        color operator*(double a)
        {
            int ared, agreen, ablue;
            ared = a*red;
            agreen = a*green;
            ablue = a*blue;
            return {(ared > 0 && ared <= 255)*ared, (agreen > 0 && agreen <= 255)*agreen, (ablue > 0 && ablue <= 255)*ablue, alpha};
        }

        int red, green, blue, alpha;
    };
}

#endif
