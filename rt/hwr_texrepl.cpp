// Texture replacements (tools/texlab.py): higher-resolution versions of the
// game's textures, keyed by the hash of the decoded pixels (R3DS_HWR_TEXDUMP
// names its dumps the same way). The executor asks for one whenever the
// recorder uploads a texture (hwr_gpu.cpp op_tex); the shaders sample in the
// texture's own texel space (textureDimensions), so any size works.
//
//   web:    the page's texture pack (textures.json + textures.pack, WebP or
//           PNG images), decoded by the browser (web/app.js, Module.r3dsTex*)
//   native: R3DS_TEX_DIR=dir of <hash>_<W>x<H>.rgba files (top row first),
//           written by texlab.py for trying replacements without packing
#include "hwr.h"
#include <string>
#include <unordered_map>
#include <mutex>
#include <cstdio>
#include <cstring>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>

EM_JS(int, r3ds_tex_query, (u32 lo, u32 hi, u32 *w, u32 *h), {
    if (!Module.r3dsTexQuery) return 0;
    return Module.r3dsTexQuery(lo >>> 0, hi >>> 0, w, h);
});
EM_JS(void, r3ds_tex_copy, (u32 lo, u32 hi, u32 *dst), {
    if (Module.r3dsTexCopy) Module.r3dsTexCopy(lo >>> 0, hi >>> 0, dst);
});

int hwr_texrepl_query(u64 hash, u32 *w, u32 *h) { return r3ds_tex_query((u32)hash, (u32)(hash >> 32), w, h); }
void hwr_texrepl_copy(u64 hash, u32 *dst) { r3ds_tex_copy((u32)hash, (u32)(hash >> 32), dst); }

#else
#include <filesystem>
namespace fsys = std::filesystem;

struct ReplFile { std::string path; u32 w, h; };
static std::unordered_map<u64, ReplFile> g_files;
static std::once_flag g_once;

static void scan() {
    const char *dir = getenv("R3DS_TEX_DIR");
    if (!dir) return;
    std::error_code ec;
    for (auto &e : fsys::directory_iterator(dir, ec)) {
        std::string n = e.path().filename().string();
        unsigned long long hash; u32 w, h;
        if (n.size() < 20 || n.substr(n.size() - 5) != ".rgba") continue;
        if (sscanf(n.c_str(), "%16llx_%ux%u", &hash, &w, &h) != 3) continue;
        if ((u64)e.file_size(ec) != (u64)w * h * 4) continue;
        g_files[hash] = {e.path().string(), w, h};
    }
    fprintf(stderr, "[hwr] %zu texture replacements in %s\n", g_files.size(), dir);
}

int hwr_texrepl_query(u64 hash, u32 *w, u32 *h) {
    std::call_once(g_once, scan);
    { static int n = 0; if (getenv("R3DS_TEXREPL_LOG") && n++ < 60) fprintf(stderr, "[texrepl] query %016llx %s\n", (unsigned long long)hash, g_files.count(hash) ? "HIT" : ""); }
    auto it = g_files.find(hash);
    if (it == g_files.end()) return 0;
    *w = it->second.w; *h = it->second.h;
    return 2;
}
void hwr_texrepl_copy(u64 hash, u32 *dst) {
    auto it = g_files.find(hash);
    if (it == g_files.end()) return;
    const u32 w = it->second.w, h = it->second.h;
    if (FILE *f = fopen(it->second.path.c_str(), "rb")) {   // top row first, as the executor wants it
        size_t n = fread(dst, 4, (size_t)w * h, f);
        fclose(f);
        (void)n;
    }
}
#endif
