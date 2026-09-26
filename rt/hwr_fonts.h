// High-resolution text for the GPU renderer.
//
// The game draws text from A4 "glyph sheets": every glyph of a font, in the
// order of the font's character table, on a grid of ((w + 1) & ~1) x (h + 1)
// cells (w x h is the font's glyph size). hwr_fonts.cpp recognises a sheet
// from its pixels and the character tables (gen/fonts_gen.cpp, generated
// from the ROM by tools/gen_fonts.py), and the executor redraws it from a
// vector font at the output resolution, fitted to each original glyph so the
// game's own layout still holds.
#pragma once
#include "common.h"
#include <vector>

struct HwrFontTable { u16 w, h, n; const u16 *codes; };
extern const HwrFontTable g_hwr_fonts[];
extern const int g_hwr_font_count;

// a recognised glyph sheet
struct HwrSheet {
    u32 cw = 0, ch = 0, cols = 0;   // cell grid (texels), cells per row
    u32 cells = 0;                  // cells of the font present in the sheet
    const HwrFontTable *font = nullptr;
};
// rgba: the decoded texture as the software sampler sees it (row t = 0 at the bottom)
bool hwr_font_detect(const u32 *rgba, u32 w, u32 h, u32 fmt, HwrSheet &out);

// glyph box of one cell, measured on the 1x sheet (memory row order: y down)
struct HwrGlyphBox { u16 code; s8 x0, y0, x1, y1; };   // ink bounds inside the cell, x1 < x0 if empty
