#include <algorithm>
#include <vs/app/app.hpp>

// Game behavior belongs to the project; VS calls it for bound keys.
struct player
{
    float altitude = .5f;
    void move_up(float dt) { altitude = std::min(1.f, altitude + dt * .5f); }
    void move_down(float dt) { altitude = std::max(0.f, altitude - dt * .5f); }
    void reset() { altitude = .5f; }
};

int main()
{
    vs::application app({.title = "vs / app - W/S move, R resets", .width = 1000, .height = 600});

    player character;

    app.on_hold('W', [&](float dt)
                { character.move_up(dt); });
    app.on_hold('S', [&](float dt)
                { character.move_down(dt); });
    app.on_press('R', [&]
                 { character.reset(); });
    app.run([&](vk::CommandBuffer cmd, std::uint32_t image)
            {
        const float h=character.altitude;
        app.clear(cmd,image,{.04f+.12f*h,.06f+.24f*h,.13f+.48f*h,1.f}); });
}
