# VulkanEditor (foundation)

A reusable, optional level-editor shell. No CSG, no scene ownership, no
GLFW/SDL dependency, no shaders, no ImGui, no game-specific object types.

## Built-in controls

- **?**: toggle editor/debug mode (host must send the typed question mark,
  not a layout-dependent physical slash key)
- **Tab**: toggle cursor freedom globally, even when editor mode is off
- **Enter**: in editor mode, open Insert; in Insert, emit selected insert request
- **Up/Down**: select Insert entries or inspector properties
- **Left/Right**: decrease/increase the highlighted property
- **Escape**: close the active menu, without leaving editor mode
- **Left mouse click** while cursor is free: select a projected world label,
  opening the object's inspector

No types or properties are included by default. The host supplies everything
that depends on its scene:

```cpp
editor::session& e=editor::instance();
e.set_catalog({{"light","Light","Lighting"},{"door","Door","Architecture"}});
e.set_labels({{42,"LIGHT",650,310,true}});
e.set_properties(42,{
    {"x","Pos X",editor::property_kind::real,11.49,-1000,1000,0.1},
    {"radius","Radius",editor::property_kind::real,4.0,0,100,0.25},
    {"strength","Strength",editor::property_kind::real,1.0,0,100,0.1}
});

// On the UI/main thread, translate platform key-down *edges* and the typed
// question-mark character into editor::input, including framebuffer pixels:
e.handle(events);
for(const auto& command:e.take_actions())
{
    // Host maps Insert to create, or Edit to its authoritative property setter.
    // No scene objects are created by the editor framework itself.
}
```

Project 3D entity positions into framebuffer **pixel coordinates** before
`set_labels`. Hide labels behind the camera/depth if required; object picking
uses the same 2D label rectangles that the overlay renders. The editor does
not know what a camera, object transform, light, or material is.

To draw, include `<editor/vulkan.hpp>` and call

```cpp
editor::vulkan::record(command, e.paint(extent.width,extent.height),extent);
```

**Only call inside an active Vulkan graphics render pass with a color attachment,**
after the scene has been drawn and before `endRenderPass`. A typical overlay
pass uses a swapchain render pass that loads its existing color attachment.
The Vulkan adapter uses `vkCmdClearAttachments` on rectangular glyph pixel
runs; it has no pipeline/shader/descriptor/texture allocation and is unaffected
by scene HDR, PBR or postprocessing. The overlay is intentionally **opaque**
(no alpha blending); it requires a color format that supports ordinary color
attachment operations. A textured/font-atlas or alpha-composited renderer can
be added later without changing `editor::session` or `draw_list`.

The host is responsible for cursor capture and gameplay input suppression,
draining the edit commands, scene synchronization, lifecycle and appropriate
Vulkan command-buffer/framebuffer ownership. Build independently:

```sh
cd editor
c build
```

`editor::session` and `draw_list` can be used without Vulkan. The packaged
`VulkanEditor` target includes the optional adapter.
