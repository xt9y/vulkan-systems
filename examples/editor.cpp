#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>
#include <vs/app/app.hpp>
#include <vs/editor/editor.hpp>
#include <vs/editor/vulkan.hpp>

// VS handles input navigation and invokes project behavior directly.
// The project owns its objects, movement rules, properties and scene.
int main()
{
    vs::application app({.title = "vs / editor - F1 to toggle"});
    vs::editor::session ui;
    struct object
    {
        std::uint64_t id = 0;

        float x = 0, y = 0;
    };

    std::vector<object> objects;
    std::uint64_t next_id = 1;

    // Register object behavior, not a list of key flags or action switches.
    ui.add("cube", "Cube", [&]
           {
        const auto id=next_id++;
        objects.push_back({id,0,0});
        ui.bind(id,{"x","X",vs::editor::property_kind::real,0,-10,10,.1},
            [&,id](double value) {
                for(auto& object:objects)if(object.id==id)object.x=float(value);
            });
        ui.bind(id,{"y","Y",vs::editor::property_kind::real,0,-10,10,.1},
            [&,id](double value) {
                for(auto& object:objects)if(object.id==id)object.y=float(value);
            }); }, "Geometry");
    ui.on_remove([&](std::uint64_t id)
                 { std::erase_if(objects, [&](const object &item)
                                 { return item.id == id; }); });

    app.attach_editor(ui); // F1, ?, arrows, Enter, Delete, mouse: all wired
    vs::editor::vulkan::initialize_font();
    app.run([&](vk::CommandBuffer cmd, std::uint32_t image)
            {
        const auto size=app.extent();
        std::vector<vs::editor::screen_label> labels;
        for(const auto& o:objects)
            labels.push_back({o.id,"CUBE "+std::to_string(o.id),
                              140+int(o.x*25),130+int(o.y*25),true});
        ui.set_labels(std::move(labels));
        app.begin_color(cmd,image,{.025f,.035f,.052f,1});
        vs::editor::vulkan::record(cmd,
            ui.paint(int(size.width),int(size.height)),size);
        app.end_color(cmd); });
    vk::device::idle();
    app.close();
}
