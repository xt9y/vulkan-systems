# VulkanEditor — Hell2024-style editor shell

Renderer-independent UI and input state, optional Vulkan sprite rasterizer. No CSG, scene, game objects or render lighting are owned by the editor.

## Original source

The canonical UI is from [Hell2025 commit 7362a3a](https://github.com/livinamuk/Hell2025/tree/7362a3a86cc254ecffdf498c9febae10eefa120b), in its `Hell2024/Hell2024/` folder. Specifically:

- `src/Editor/Editor.cpp`: menu row formatting, background, border/corners, selection, drag logic
- `src/Renderer/TextBlitting.cpp`: original bitmap-font advances, 16px lines, byte-indexed character sheet
- `res/textures/font/char_1.png` through `char_90.png`: the original bitmap font
- `res/textures/ui/MenuBG.png`, `MenuBorder*.png`, and `Icon_Light.jpg`: the original menu and world-label images

The original UI uses a 960×540 presentation canvas. The Vulkan adapter scales its output to the swapchain framebuffer. There is no permanent sidebar. `?` reveals world labels, while the framed floating menu only appears on Insert or object selection.

Floating menus use the original 200px logical width and 16px row spacing, a centered dashed heading, selection markers (`>` for items, `<value>` for properties), and the actual original textures. The menu height is held to 245 logical pixels in both modes, with selection scrolling, per the host game's requirements.

## Controls

- `?`: toggle editor mode; show host-projected world object labels
- `Tab`: always toggle free/captured cursor
- `Enter`: open the Insert menu or confirm a selected host-defined item
- `Up`/`Down`: navigate and scroll menu entries
- `Left`/`Right`: edit selected object's host-defined property
- `Escape`: close the current menu without leaving editor mode
- Left-click a projected label while cursor is free: open its properties
- Drag menu header with the released mouse cursor to reposition the panel

## Host integration

The host supplies all object types, names, labels, property definitions and the handlers for insert/edit commands. These are not hardcoded. For example:

```cpp
auto& ui=editor::instance();
ui.set_catalog({{"light","Light"},{"door","Door"}});
ui.set_labels({{42,"LIGHT",screen_x,screen_y,true}});
ui.set_properties(42,{
    {"radius","Radius",editor::property_kind::real,4.0,0.0,100.0,0.25},
    {"strength","Strength",editor::property_kind::real,1.0,0.0,100.0,0.1}
});
ui.handle(input_frame);
for(const auto& change:ui.take_actions()) { /* host updates its scene */ }
```

Label coordinates are framebuffer pixels, converted into the original 960×540 editor canvas by the UI. The host must mark invisible or behind-camera objects as hidden.

## Bitmap font and UI assets

The user-supplied `x_font.png` is now embedded in `editor/editor/uploaded_font.hpp` as a **pixel-exact 144×256 bitmap** (16×16 CP437 grid, 9×16 pixels per character). ASCII characters map directly to their codepoint cell. The atlas is decoded once at startup; the old Hell2024 `char_*.png` font is no longer used or downloaded. There is no synthetic fallback font.

The game still fetches the original Hell2024 **UI textures only** from the pinned Hell2025 revision into `build/editor-hell2024/Hell2024/Hell2024/res/textures/ui/`. All eight UI textures remain mandatory.

Initialize once on the startup thread, before Vulkan rendering begins:

```cpp
editor::vulkan::initialize_font();
editor::instance().set_font_metrics(editor::vulkan::font_metrics(),16);
```

This decodes the embedded font atlas and the original UI textures, then caches drawing spans. Neither font loading nor filesystem access occurs during an editor frame. Missing UI textures cause an explicit startup error.

Draw in the existing presentation render pass, after the scene is tonemapped:

```cpp
editor::vulkan::record(cmd,ui.paint(extent.width,extent.height),extent);
```

The default backend uses `vkCmdClearAttachments`, avoiding any dependency on scene shaders, GI, PBR, or render targets. This reproduces the original layout and samples from the actual original textures, but **does not provide the original alpha blending or filtering**. Semi-transparent texture pixels are approximated; a specialized alpha-blended Vulkan backend could later replace only this adapter without rewriting the editor session.

The host remains responsible for input event translation, cursor capture, scene changes, viewport projection and Vulkan synchronization.
