// SDL2 frontend: window, keyboard / gamepad / mouse-as-touch input.
// Runs on the process main thread; guest threads hand frames over through
// display_get_frame().
#include "common.h"
#include "platform.h"
#include "input.h"
#include <atomic>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstring>
#ifdef R3DS_HAVE_SDL
#define SDL_MAIN_HANDLED
#include <SDL.h>
#include <SDL_syswm.h>
#ifdef __APPLE__
#include <SDL_metal.h>
#endif
#endif
#include "hwr.h"

bool display_get_frame(std::vector<u32> &out);   // display.cpp
void dsp_audio_pull(s16 *dst, int frames);        // hle_dsp_audio.cpp

#ifdef R3DS_HAVE_SDL
float display_bottom_coverage();                  // display.cpp
bool display_bottom_masked();

static const char *HELP =
    "[sdl] controls: D-pad=arrows  Circle pad=WASD  A=K  B=J  X=I  Y=U  L=Q  R=E  Start=Enter  Select=Backspace\n"
    "[sdl]           touch=mouse on the bottom screen  (gamepads supported)\n"
    "[sdl] display:  F1=layout (remaster/classic/side/top)  F2=bottom mode (auto/ui/minimap)  F3=remove bottom background\n"
    "[sdl]           F4=filter (smooth/pixel)  F5=aspect (5:3 / 16:9)  F6=side bars (blur/black)  hold Tab=whole bottom screen\n"
    "[sdl]           F11=fullscreen  F12=screenshot  Esc=quit\n";

// Screen layout, mirroring web/compositor.js. The bottom screen's alpha is the
// renderer's content mask (0 = backdrop), so in the remaster layout it floats
// over the top screen with its background removed.
enum Layout { L_REMASTER, L_CLASSIC, L_SIDE, L_TOP, L_COUNT };
enum BottomMode { B_AUTO, B_UI, B_MINIMAP, B_COUNT };
static const char *LAYOUT_NAMES[] = {"remaster", "classic", "side", "top"};
static const char *BMODE_NAMES[] = {"auto", "ui", "minimap"};
struct Opts {
    int layout = L_REMASTER, bmode = B_AUTO;
    bool remove_bg = true, smooth = true, wide = false, blur = true;
    float ui_scale = 0.85f, ui_alpha = 0.96f, mini_scale = 0.36f, mini_alpha = 0.9f;
};
struct FRect { float x, y, w, h; };
static FRect fit(FRect a, float aspect) {
    float w = a.w, h = w / aspect;
    if (h > a.h) { h = a.h; w = h * aspect; }
    return {a.x + (a.w - w) / 2, a.y + (a.h - h) / 2, w, h};
}
static SDL_Rect irect(const FRect &r) {
    return {(int)lroundf(r.x), (int)lroundf(r.y), (int)lroundf(r.x + r.w) - (int)lroundf(r.x), (int)lroundf(r.y + r.h) - (int)lroundf(r.y)};
}
static int pick(const char *env, const char *const *names, int n, int def) {
    const char *v = getenv(env);
    if (v) for (int i = 0; i < n; i++) if (!strcmp(v, names[i])) return i;
    return def;
}

// shared by both display paths: keyboard / gamepad state -> InputState (no touch)
// swap: WASD drive the D-pad and the arrows the circle pad (menus on the bottom screen)
static void read_pad(SDL_GameController *pad, InputState &in, bool swap = false) {
    const Uint8 *k = SDL_GetKeyboardState(nullptr);
    auto key = [&](SDL_Scancode s) {
        if (swap) {
            switch (s) {
            case SDL_SCANCODE_UP: s = SDL_SCANCODE_W; break;       case SDL_SCANCODE_W: s = SDL_SCANCODE_UP; break;
            case SDL_SCANCODE_DOWN: s = SDL_SCANCODE_S; break;     case SDL_SCANCODE_S: s = SDL_SCANCODE_DOWN; break;
            case SDL_SCANCODE_LEFT: s = SDL_SCANCODE_A; break;     case SDL_SCANCODE_A: s = SDL_SCANCODE_LEFT; break;
            case SDL_SCANCODE_RIGHT: s = SDL_SCANCODE_D; break;    case SDL_SCANCODE_D: s = SDL_SCANCODE_RIGHT; break;
            default: break;
            }
        }
        return k[s] != 0;
    };
    if (key(SDL_SCANCODE_K)) in.buttons |= BTN_A;
    if (key(SDL_SCANCODE_J)) in.buttons |= BTN_B;
    if (key(SDL_SCANCODE_I)) in.buttons |= BTN_X;
    if (key(SDL_SCANCODE_U)) in.buttons |= BTN_Y;
    if (key(SDL_SCANCODE_Q)) in.buttons |= BTN_L;
    if (key(SDL_SCANCODE_E)) in.buttons |= BTN_R;
    if (key(SDL_SCANCODE_RETURN)) in.buttons |= BTN_START;
    if (key(SDL_SCANCODE_BACKSPACE)) in.buttons |= BTN_SELECT;
    if (key(SDL_SCANCODE_UP)) in.buttons |= BTN_DUP;
    if (key(SDL_SCANCODE_DOWN)) in.buttons |= BTN_DDOWN;
    if (key(SDL_SCANCODE_LEFT)) in.buttons |= BTN_DLEFT;
    if (key(SDL_SCANCODE_RIGHT)) in.buttons |= BTN_DRIGHT;
    int cx = 0, cy = 0;
    if (key(SDL_SCANCODE_A)) cx -= 156;
    if (key(SDL_SCANCODE_D)) cx += 156;
    if (key(SDL_SCANCODE_W)) cy += 156;
    if (key(SDL_SCANCODE_S)) cy -= 156;
    if (pad) {
        auto b = [&](SDL_GameControllerButton x) { return SDL_GameControllerGetButton(pad, x) != 0; };
        if (b(SDL_CONTROLLER_BUTTON_B)) in.buttons |= BTN_A;       // positional (Nintendo layout)
        if (b(SDL_CONTROLLER_BUTTON_A)) in.buttons |= BTN_B;
        if (b(SDL_CONTROLLER_BUTTON_Y)) in.buttons |= BTN_X;
        if (b(SDL_CONTROLLER_BUTTON_X)) in.buttons |= BTN_Y;
        if (b(SDL_CONTROLLER_BUTTON_LEFTSHOULDER)) in.buttons |= BTN_L;
        if (b(SDL_CONTROLLER_BUTTON_RIGHTSHOULDER)) in.buttons |= BTN_R;
        if (b(SDL_CONTROLLER_BUTTON_START)) in.buttons |= BTN_START;
        if (b(SDL_CONTROLLER_BUTTON_BACK)) in.buttons |= BTN_SELECT;
        if (b(SDL_CONTROLLER_BUTTON_DPAD_UP)) in.buttons |= BTN_DUP;
        if (b(SDL_CONTROLLER_BUTTON_DPAD_DOWN)) in.buttons |= BTN_DDOWN;
        if (b(SDL_CONTROLLER_BUTTON_DPAD_LEFT)) in.buttons |= BTN_DLEFT;
        if (b(SDL_CONTROLLER_BUTTON_DPAD_RIGHT)) in.buttons |= BTN_DRIGHT;
        int ax = SDL_GameControllerGetAxis(pad, SDL_CONTROLLER_AXIS_LEFTX);
        int ay = SDL_GameControllerGetAxis(pad, SDL_CONTROLLER_AXIS_LEFTY);
        if (abs(ax) > 6000) cx = ax * 156 / 32767;
        if (abs(ay) > 6000) cy = -ay * 156 / 32767;
    }
    in.cx = (s16)cx; in.cy = (s16)cy;
    // circle pad also drives the "circle pad direction" button bits
    if (cx > 80) in.buttons |= 1u << 28;
    if (cx < -80) in.buttons |= 1u << 29;
    if (cy > 80) in.buttons |= 1u << 30;
    if (cy < -80) in.buttons |= 1u << 31;
}

static void open_audio() {
    SDL_AudioSpec want{}, have{};
    want.freq = 32728; want.format = AUDIO_S16SYS; want.channels = 2; want.samples = 512;
    want.callback = [](void *, Uint8 *stream, int len) { dsp_audio_pull((s16 *)stream, len / 4); };
    SDL_AudioDeviceID adev = SDL_OpenAudioDevice(nullptr, 0, &want, &have, 0);
    if (adev) SDL_PauseAudioDevice(adev, 0);
    else LOG("[sdl] no audio device: %s", SDL_GetError());
}

#ifdef R3DS_WEBGPU
// ---- WebGPU path: the hardware renderer draws and composes the screens
static const char *GPU_HELP =
    "[sdl] controls: D-pad=arrows  Circle pad=WASD  A=K  B=J  X=I  Y=U  L=Q  R=E  Start=Enter  Select=Backspace\n"
    "[sdl]           touch=mouse on the bottom screen  (gamepads supported)\n"
    "[sdl] display:  F1=layout (remaster/classic/side/top/large)  F2=bottom mode  F3=remove bottom background  F4=filter\n"
    "[sdl]           F5=aspect (5:3/auto/16:9/stretch)  F6=side bars  F7=resolution (auto/1x..6x)  F8=bottom screen position\n"
    "[sdl]           F9/F10=bottom screen size  hold Tab=whole bottom screen  F11=fullscreen  F12=screenshot  Esc=quit\n"
    "[sdl]           while the bottom screen shows a menu, WASD and the arrows swap (R3DS_SWAP_KEYS=0 to keep them)\n";
static bool gpu_target(SDL_Window *win, HwrSurfaceTarget &t) {
#ifdef __APPLE__
    SDL_MetalView mv = SDL_Metal_CreateView(win);
    if (!mv) return false;
    t.kind = HWT_METAL; t.a = SDL_Metal_GetLayer(mv);
    return t.a != nullptr;
#else
    SDL_SysWMinfo wm;
    SDL_VERSION(&wm.version);
    if (!SDL_GetWindowWMInfo(win, &wm)) return false;
#if defined(_WIN32)
    if (wm.subsystem == SDL_SYSWM_WINDOWS) { t.kind = HWT_WIN32; t.a = wm.info.win.hinstance; t.b = wm.info.win.window; return true; }
#else
#if defined(SDL_VIDEO_DRIVER_WAYLAND)
    if (wm.subsystem == SDL_SYSWM_WAYLAND) { t.kind = HWT_WAYLAND; t.a = wm.info.wl.display; t.b = wm.info.wl.surface; return true; }
#endif
#if defined(SDL_VIDEO_DRIVER_X11)
    if (wm.subsystem == SDL_SYSWM_X11) { t.kind = HWT_X11; t.a = wm.info.x11.display; t.b = (void *)(uintptr_t)wm.info.x11.window; return true; }
#endif
#endif
    return false;
#endif
}
static void drawable_size(SDL_Window *win, int &w, int &h) {
#ifdef __APPLE__
    SDL_Metal_GetDrawableSize(win, &w, &h);
#elif SDL_VERSION_ATLEAST(2, 26, 0)
    SDL_GetWindowSizeInPixels(win, &w, &h);
#else
    SDL_GetWindowSize(win, &w, &h);
#endif
}
static void run_gpu(SDL_Window *win) {
    HwrView v = hwr_view_from_env();
    static const char *LAY[] = {"remaster", "classic (Citra default)", "side", "top", "large (Citra)"}, *BM[] = {"auto", "ui", "minimap"},
                      *ASP[] = {"5:3", "auto (window)", "16:9", "16:9 stretched"}, *FIL[] = {"sharp pixels", "bilinear", "bicubic"},
                      *POS[] = {"bottom right", "bottom centre", "bottom left", "top right", "top left"};
    auto status = [&]() {
        LOG("[sdl] layout=%s bottom=%s remove_bg=%d filter=%s aspect=%s bars=%d resolution=%s%d position=%s size=%.2f",
            LAY[v.layout], BM[v.bottom_mode], v.remove_bg, FIL[v.filter], ASP[v.aspect], v.side_blur,
            v.res ? "" : "auto/", v.res ? v.res : (int)g_hwr_req_scale.load(), POS[v.ui_anchor], v.ui_scale);
    };
    fputs(GPU_HELP, stderr);
    status();
    open_audio();
    SDL_GameController *pad = nullptr;
    bool mouse_down = false;
    HwrOut out;
    int shot = 0;
    u32 frames = 0; u64 t0 = SDL_GetTicks64();
    for (;;) {
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) { SDL_Quit(); plat_exit(0); }
            if (e.type == SDL_CONTROLLERDEVICEADDED && !pad) pad = SDL_GameControllerOpen(e.cdevice.which);
            if (e.type == SDL_KEYDOWN && !e.key.repeat) {
                switch (e.key.keysym.sym) {
                case SDLK_ESCAPE: SDL_Quit(); plat_exit(0); break;
                case SDLK_F1: v.layout = (v.layout + 1) % 5; status(); break;
                case SDLK_F2: v.bottom_mode = (v.bottom_mode + 1) % 3; status(); break;
                case SDLK_F3: v.remove_bg = !v.remove_bg; status(); break;
                case SDLK_F4: v.filter = (v.filter + 1) % 3; status(); break;
                case SDLK_F5: v.aspect = (v.aspect + 1) % 4; status(); break;
                case SDLK_F6: v.side_blur = !v.side_blur; status(); break;
                case SDLK_F7: v.res = (v.res + 1) % 7; status(); break;
                case SDLK_F8: v.ui_anchor = (v.ui_anchor + 1) % 5; status(); break;
                case SDLK_F9: v.ui_scale = std::max(0.3f, v.ui_scale - 0.05f); status(); break;
                case SDLK_F10: v.ui_scale = std::min(1.0f, v.ui_scale + 0.05f); status(); break;
                case SDLK_F11:
                    SDL_SetWindowFullscreen(win, (SDL_GetWindowFlags(win) & SDL_WINDOW_FULLSCREEN_DESKTOP) ? 0 : SDL_WINDOW_FULLSCREEN_DESKTOP);
                    break;
                case SDLK_F12: {
                    char path[64]; snprintf(path, sizeof path, "screenshot_%03d.bmp", shot++);
                    std::vector<u32> opaque(400 * 480);
                    for (int i = 0; i < 400 * 480; i++) opaque[i] = hwr_gpu_soft_pixel(i % 400, i / 400) | 0xFF000000u;
                    SDL_Surface *s = SDL_CreateRGBSurfaceWithFormatFrom(opaque.data(), 400, 480, 32, 1600, SDL_PIXELFORMAT_ABGR8888);
                    SDL_SaveBMP(s, path); SDL_FreeSurface(s);
                    LOG("[sdl] saved %s (3DS resolution)", path);
                    break;
                }
                default: break;
                }
            }
            if (e.type == SDL_MOUSEBUTTONDOWN && e.button.button == SDL_BUTTON_LEFT) mouse_down = true;
            if (e.type == SDL_MOUSEBUTTONUP && e.button.button == SDL_BUTTON_LEFT) mouse_down = false;
        }
        InputState in{};
        static bool swap_keys = !getenv("R3DS_SWAP_KEYS") || strcmp(getenv("R3DS_SWAP_KEYS"), "0") != 0;
        read_pad(pad, in, swap_keys && out.bottom_focus);
        v.show_full = SDL_GetKeyboardState(nullptr)[SDL_SCANCODE_TAB] != 0;
        int W, H, ww, wh;
        drawable_size(win, W, H);
        SDL_GetWindowSize(win, &ww, &wh);
        if (mouse_down && out.bottom_on && out.bw > 0) {
            int mx, my;
            SDL_GetMouseState(&mx, &my);
            float px = mx * (float)W / ww, py = my * (float)H / wh;   // window -> drawable pixels (HiDPI)
            int tx = (int)floorf((px - out.bx) * 320 / out.bw), ty = (int)floorf((py - out.by) * 240 / out.bh);
            bool inside = tx >= 0 && tx < 320 && ty >= 0 && ty < 240;
            // clicks on the removed background go to nobody, so they don't press hidden buttons
            if (inside && out.hit && (hwr_gpu_soft_pixel(40 + tx, 240 + ty) >> 24) < 24) inside = false;
            if (inside) { in.touch = 1; in.tx = tx; in.ty = ty; }
        }
        input_set(in);
        bool shown = hwr_gpu_frame((u32)W, (u32)H, v, out, false);
        if (shown) frames++;
        else SDL_Delay(1);
        u64 t = SDL_GetTicks64();
        if (t - t0 >= 5000) {
            char title[128];
            snprintf(title, sizeof title, "recomp3ds - %.1f fps - %dx internal%s", frames * 1000.0 / (t - t0),
                     (int)g_hwr_req_scale.load(), out.top_gpu ? "" : " (software image)");
            SDL_SetWindowTitle(win, title);
            LOG("[sdl] %.1f fps, %dx%d window, %dx internal resolution, top %s, bottom %s, bottom coverage %.3f", frames * 1000.0 / (t - t0), W, H,
                (int)g_hwr_req_scale.load(), out.top_gpu ? "gpu" : "software", out.bot_gpu ? "gpu" : "software", display_bottom_coverage());
            frames = 0; t0 = t;
        }
    }
}
#endif

bool frontend_run() {
    if (getenv("R3DS_HEADLESS")) return false;
#if !defined(_WIN32) && !defined(__APPLE__)   // macOS has no DISPLAY
    if (!getenv("DISPLAY") && !getenv("WAYLAND_DISPLAY")) return false;
#endif
    SDL_SetMainReady();
    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_GAMECONTROLLER | SDL_INIT_AUDIO) != 0) {
        LOG("[sdl] init failed: %s (running headless)", SDL_GetError());
        return false;
    }
    Opts O;
    O.layout = pick("R3DS_LAYOUT", LAYOUT_NAMES, L_COUNT, L_REMASTER);
    O.bmode = pick("R3DS_BOTTOM", BMODE_NAMES, B_COUNT, B_AUTO);
    if (const char *v = getenv("R3DS_FILTER")) O.smooth = strcmp(v, "pixel") != 0;
    if (const char *v = getenv("R3DS_ASPECT")) O.wide = !strcmp(v, "16:9") || !strcmp(v, "stretch");
    if (getenv("R3DS_KEEP_BG")) O.remove_bg = false;
    int scale = getenv("R3DS_SCALE") ? atoi(getenv("R3DS_SCALE")) : 2;
    int ww = O.layout == L_CLASSIC ? 400 * scale : (O.layout == L_SIDE ? 592 * scale : 400 * scale);
    int wh = O.layout == L_CLASSIC ? 480 * scale : 240 * scale;
#ifdef R3DS_WEBGPU
    // WebGPU renderer unless R3DS_RENDERER=soft (or it cannot start)
    if (!getenv("R3DS_RENDERER") || strcmp(getenv("R3DS_RENDERER"), "soft") != 0) {
        SDL_DisplayMode dm;
        int gw = 1280, gh = 720;
        if (!getenv("R3DS_SCALE") && SDL_GetDesktopDisplayMode(0, &dm) == 0 && dm.w > 0) { gw = dm.w * 3 / 4; gh = gw * 9 / 16; }
        else if (getenv("R3DS_SCALE")) { gw = 400 * scale; gh = 240 * scale; }
        Uint32 flags = SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI;
#ifdef __APPLE__
        flags |= SDL_WINDOW_METAL;
#endif
        SDL_Window *gwin = SDL_CreateWindow("recomp3ds", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, gw, gh, flags);
        HwrSurfaceTarget t;
        if (gwin && gpu_target(gwin, t) && hwr_gpu_start(t) && hwr_gpu_state() == 2) {
            if (getenv("R3DS_FULLSCREEN")) SDL_SetWindowFullscreen(gwin, SDL_WINDOW_FULLSCREEN_DESKTOP);
            run_gpu(gwin);
        }
        LOG("[sdl] WebGPU renderer unavailable: using the software display");
        if (gwin) SDL_DestroyWindow(gwin);
    }
#endif
    SDL_Window *win = SDL_CreateWindow("recomp3ds", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                       ww, wh, SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI);
    SDL_Renderer *ren = SDL_CreateRenderer(win, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC | SDL_RENDERER_TARGETTEXTURE);
    if (!ren) ren = SDL_CreateRenderer(win, -1, 0);
    SDL_Texture *ttop = SDL_CreateTexture(ren, SDL_PIXELFORMAT_ABGR8888, SDL_TEXTUREACCESS_STREAMING, 400, 240);
    SDL_Texture *tbot = SDL_CreateTexture(ren, SDL_PIXELFORMAT_ABGR8888, SDL_TEXTUREACCESS_STREAMING, 320, 240);   // alpha = mask
    SDL_Texture *tbotop = SDL_CreateTexture(ren, SDL_PIXELFORMAT_ABGR8888, SDL_TEXTUREACCESS_STREAMING, 320, 240); // opaque
    // side-bar blur: the top screen scaled down into a tiny target, then stretched back up
    SDL_Texture *tblur = SDL_RenderTargetSupported(ren)
        ? SDL_CreateTexture(ren, SDL_PIXELFORMAT_ABGR8888, SDL_TEXTUREACCESS_TARGET, 40, 24) : nullptr;
    if (tblur) { SDL_SetTextureScaleMode(tblur, SDL_ScaleModeLinear); SDL_SetTextureColorMod(tblur, 110, 110, 110); }
    fputs(HELP, stderr);
    SDL_AudioSpec want{}, have{};
    want.freq = 32728; want.format = AUDIO_S16SYS; want.channels = 2; want.samples = 512;
    want.callback = [](void *, Uint8 *stream, int len) { dsp_audio_pull((s16 *)stream, len / 4); };
    SDL_AudioDeviceID adev = SDL_OpenAudioDevice(nullptr, 0, &want, &have, 0);
    if (adev) SDL_PauseAudioDevice(adev, 0);
    else LOG("[sdl] no audio device: %s", SDL_GetError());
    std::vector<u32> frame(400 * 480, 0xFF000000u), bot(320 * 240, 0), botop(320 * 240, 0);
    SDL_GameController *pad = nullptr;
    bool mouse_down = false, have_frame = false, masked = false;
    float coverage = 1.0f;
    int mode = B_UI, votes = 0;          // automatic UI / minimap choice, with hysteresis
    FRect cur{0, 0, 0, 0}, target{0, 0, 0, 0}; bool have_cur = false;   // animated bottom rect
    FRect bot_rect{0, 0, 0, 0}; bool bot_on = false, bot_hit = false;
    int shot = 0;
    auto status = [&]() {
        LOG("[sdl] layout=%s bottom=%s remove_bg=%d filter=%s aspect=%s bars=%s", LAYOUT_NAMES[O.layout], BMODE_NAMES[O.bmode],
            O.remove_bg, O.smooth ? "smooth" : "pixel", O.wide ? "16:9" : "5:3", O.blur ? "blur" : "black");
    };
    status();
    for (;;) {
        bool dirty = false;
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) { SDL_Quit(); plat_exit(0); }
            if (e.type == SDL_CONTROLLERDEVICEADDED && !pad) pad = SDL_GameControllerOpen(e.cdevice.which);
            if (e.type == SDL_WINDOWEVENT) dirty = true;
            if (e.type == SDL_KEYDOWN && !e.key.repeat) {
                switch (e.key.keysym.sym) {
                case SDLK_ESCAPE: SDL_Quit(); plat_exit(0); break;
                case SDLK_F1: O.layout = (O.layout + 1) % L_COUNT; have_cur = false; status(); break;
                case SDLK_F2: O.bmode = (O.bmode + 1) % B_COUNT; status(); break;
                case SDLK_F3: O.remove_bg = !O.remove_bg; status(); break;
                case SDLK_F4: O.smooth = !O.smooth; status(); break;
                case SDLK_F5: O.wide = !O.wide; status(); break;
                case SDLK_F6: O.blur = !O.blur; status(); break;
                case SDLK_F11:
                    SDL_SetWindowFullscreen(win, (SDL_GetWindowFlags(win) & SDL_WINDOW_FULLSCREEN_DESKTOP) ? 0 : SDL_WINDOW_FULLSCREEN_DESKTOP);
                    break;
                case SDLK_F12: {
                    char path[64]; snprintf(path, sizeof path, "screenshot_%03d.bmp", shot++);
                    std::vector<u32> opaque(frame);
                    for (u32 &px : opaque) px |= 0xFF000000u;
                    SDL_Surface *s = SDL_CreateRGBSurfaceWithFormatFrom(opaque.data(), 400, 480, 32, 1600, SDL_PIXELFORMAT_ABGR8888);
                    SDL_SaveBMP(s, path); SDL_FreeSurface(s);
                    LOG("[sdl] saved %s", path);
                    break;
                }
                default: break;
                }
                dirty = true;
            }
            if (e.type == SDL_KEYUP && e.key.keysym.sym == SDLK_TAB) dirty = true;
            if (e.type == SDL_MOUSEBUTTONDOWN && e.button.button == SDL_BUTTON_LEFT) mouse_down = true;
            if (e.type == SDL_MOUSEBUTTONUP && e.button.button == SDL_BUTTON_LEFT) mouse_down = false;
        }
        InputState in{};
        const Uint8 *k = SDL_GetKeyboardState(nullptr);
        auto key = [&](SDL_Scancode s) { return k[s] != 0; };
        const bool show_full = key(SDL_SCANCODE_TAB);
        if (key(SDL_SCANCODE_K)) in.buttons |= BTN_A;
        if (key(SDL_SCANCODE_J)) in.buttons |= BTN_B;
        if (key(SDL_SCANCODE_I)) in.buttons |= BTN_X;
        if (key(SDL_SCANCODE_U)) in.buttons |= BTN_Y;
        if (key(SDL_SCANCODE_Q)) in.buttons |= BTN_L;
        if (key(SDL_SCANCODE_E)) in.buttons |= BTN_R;
        if (key(SDL_SCANCODE_RETURN)) in.buttons |= BTN_START;
        if (key(SDL_SCANCODE_BACKSPACE)) in.buttons |= BTN_SELECT;
        if (key(SDL_SCANCODE_UP)) in.buttons |= BTN_DUP;
        if (key(SDL_SCANCODE_DOWN)) in.buttons |= BTN_DDOWN;
        if (key(SDL_SCANCODE_LEFT)) in.buttons |= BTN_DLEFT;
        if (key(SDL_SCANCODE_RIGHT)) in.buttons |= BTN_DRIGHT;
        int cx = 0, cy = 0;
        if (key(SDL_SCANCODE_A)) cx -= 156;
        if (key(SDL_SCANCODE_D)) cx += 156;
        if (key(SDL_SCANCODE_W)) cy += 156;
        if (key(SDL_SCANCODE_S)) cy -= 156;
        if (pad) {
            auto b = [&](SDL_GameControllerButton x) { return SDL_GameControllerGetButton(pad, x) != 0; };
            if (b(SDL_CONTROLLER_BUTTON_B)) in.buttons |= BTN_A;       // positional (Nintendo layout)
            if (b(SDL_CONTROLLER_BUTTON_A)) in.buttons |= BTN_B;
            if (b(SDL_CONTROLLER_BUTTON_Y)) in.buttons |= BTN_X;
            if (b(SDL_CONTROLLER_BUTTON_X)) in.buttons |= BTN_Y;
            if (b(SDL_CONTROLLER_BUTTON_LEFTSHOULDER)) in.buttons |= BTN_L;
            if (b(SDL_CONTROLLER_BUTTON_RIGHTSHOULDER)) in.buttons |= BTN_R;
            if (b(SDL_CONTROLLER_BUTTON_START)) in.buttons |= BTN_START;
            if (b(SDL_CONTROLLER_BUTTON_BACK)) in.buttons |= BTN_SELECT;
            if (b(SDL_CONTROLLER_BUTTON_DPAD_UP)) in.buttons |= BTN_DUP;
            if (b(SDL_CONTROLLER_BUTTON_DPAD_DOWN)) in.buttons |= BTN_DDOWN;
            if (b(SDL_CONTROLLER_BUTTON_DPAD_LEFT)) in.buttons |= BTN_DLEFT;
            if (b(SDL_CONTROLLER_BUTTON_DPAD_RIGHT)) in.buttons |= BTN_DRIGHT;
            int ax = SDL_GameControllerGetAxis(pad, SDL_CONTROLLER_AXIS_LEFTX);
            int ay = SDL_GameControllerGetAxis(pad, SDL_CONTROLLER_AXIS_LEFTY);
            if (abs(ax) > 6000) cx = ax * 156 / 32767;
            if (abs(ay) > 6000) cy = -ay * 156 / 32767;
        }
        in.cx = (s16)cx; in.cy = (s16)cy;
        // circle pad also drives the "circle pad direction" button bits
        if (cx > 80) in.buttons |= 1u << 28;
        if (cx < -80) in.buttons |= 1u << 29;
        if (cy > 80) in.buttons |= 1u << 30;
        if (cy < -80) in.buttons |= 1u << 31;
        if (mouse_down && bot_on) {
            int mx, my, winw, winh, outw, outh;
            SDL_GetMouseState(&mx, &my);
            SDL_GetWindowSize(win, &winw, &winh);
            SDL_GetRendererOutputSize(ren, &outw, &outh);
            float px = mx * (float)outw / winw, py = my * (float)outh / winh;   // window -> output pixels (HiDPI)
            int tx = (int)floorf((px - bot_rect.x) * 320 / bot_rect.w), ty = (int)floorf((py - bot_rect.y) * 240 / bot_rect.h);
            bool inside = tx >= 0 && tx < 320 && ty >= 0 && ty < 240;
            // clicks on the removed background go to nobody, so they don't press hidden buttons
            if (inside && bot_hit && (bot[ty * 320 + tx] >> 24) < 24) inside = false;
            if (inside) { in.touch = 1; in.tx = tx; in.ty = ty; }
        }
        input_set(in);

        if (display_get_frame(frame)) {
            have_frame = dirty = true;
            masked = display_bottom_masked();
            coverage = display_bottom_coverage();
            for (int y = 0; y < 240; y++)
                for (int x = 0; x < 320; x++) {
                    u32 p = frame[(240 + y) * 400 + 40 + x];
                    bot[y * 320 + x] = masked ? p : (p | 0xFF000000u);
                    botop[y * 320 + x] = p | 0xFF000000u;
                }
            SDL_UpdateTexture(ttop, nullptr, frame.data(), 400 * 4);
            SDL_UpdateTexture(tbot, nullptr, bot.data(), 320 * 4);
            SDL_UpdateTexture(tbotop, nullptr, botop.data(), 320 * 4);
        }
        static bool last_full = false;
        if (show_full != last_full) { last_full = show_full; dirty = true; }
        const bool animating = have_cur && (fabsf(cur.w - target.w) > 0.5f || fabsf(cur.x - target.x) > 0.5f || fabsf(cur.y - target.y) > 0.5f);
        if (!have_frame || (!dirty && !animating)) { SDL_Delay(2); continue; }

        // ---- layout (output pixels)
        int W, H;
        SDL_GetRendererOutputSize(ren, &W, &H);
        const FRect full{0, 0, (float)W, (float)H};
        const float top_aspect = O.wide ? 16.0f / 9.0f : 5.0f / 3.0f;
        FRect top, br{0, 0, 0, 0};
        bool bon = true, premul = false;
        float balpha = 1.0f;
        if (O.layout == L_CLASSIC) {
            FRect r = fit(full, 400.0f / 480.0f);
            float s = r.w / 400;
            top = {r.x, r.y, r.w, 240 * s};
            br = {r.x + 40 * s, r.y + 240 * s, 320 * s, 240 * s};
        } else if (O.layout == L_SIDE) {
            FRect r = fit(full, (400 + 320 * 0.6f) / 240);
            float s = r.h / 240, bs = s * 0.6f;
            top = {r.x, r.y, 400 * s, 240 * s};
            br = {r.x + 400 * s, r.y + 240 * s - 240 * bs, 320 * bs, 240 * bs};
        } else {
            top = fit(full, top_aspect);
            float s = top.h / 240;
            int want = O.bmode;
            if (want == B_AUTO) {
                int guess = masked && coverage > 0.72f ? B_MINIMAP : B_UI;
                if (guess != mode) { if (++votes > 20) { mode = guess; votes = 0; } } else votes = 0;
                want = mode;
            }
            premul = O.remove_bg && masked;
            if (show_full) {
                float bs = s * std::max(O.ui_scale, 0.9f);
                br = {top.x + (top.w - 320 * bs) / 2, top.y + top.h - 240 * bs, 320 * bs, 240 * bs};
                premul = false;
            } else if (O.layout == L_TOP) {
                bon = false;
            } else if (want == B_MINIMAP) {
                float bs = s * O.mini_scale, m = 12;
                br = {top.x + top.w - 320 * bs - m, top.y + top.h - 240 * bs - m, 320 * bs, 240 * bs};
                balpha = O.mini_alpha;
            } else {
                float bs = s * O.ui_scale;
                br = {top.x + (top.w - 320 * bs) / 2, top.y + top.h - 240 * bs, 320 * bs, 240 * bs};
                balpha = O.ui_alpha;
            }
            target = br;
            if (bon && have_cur && !show_full) {   // animate between placements
                const float kk = 0.25f;
                cur.x += (br.x - cur.x) * kk; cur.y += (br.y - cur.y) * kk;
                cur.w += (br.w - cur.w) * kk; cur.h += (br.h - cur.h) * kk;
                br = cur;
            } else if (bon) { cur = br; }
            have_cur = bon && !show_full;
        }

        // ---- draw
        const SDL_ScaleMode sm = O.smooth ? SDL_ScaleModeLinear : SDL_ScaleModeNearest;
        SDL_SetTextureScaleMode(ttop, sm);
        SDL_SetTextureScaleMode(tbot, sm);
        SDL_SetRenderDrawColor(ren, 0, 0, 0, 255);
        SDL_RenderClear(ren);
        SDL_Rect topsrc{0, 0, 400, 240};
        if (O.blur && tblur && O.layout != L_CLASSIC && (top.x > 1 || top.y > 1)) {
            SDL_SetRenderTarget(ren, tblur);
            SDL_SetTextureBlendMode(ttop, SDL_BLENDMODE_NONE);
            SDL_SetTextureScaleMode(ttop, SDL_ScaleModeLinear);
            SDL_RenderCopy(ren, ttop, &topsrc, nullptr);
            SDL_SetRenderTarget(ren, nullptr);
            SDL_SetTextureScaleMode(ttop, sm);
            FRect cover{0, 0, (float)W, W / top_aspect};
            if (cover.h < H) { cover.h = (float)H; cover.w = H * top_aspect; }
            cover.x = (W - cover.w) / 2; cover.y = (H - cover.h) / 2;
            SDL_Rect cr = irect(cover);
            SDL_RenderCopy(ren, tblur, nullptr, &cr);
        }
        SDL_SetTextureBlendMode(ttop, SDL_BLENDMODE_NONE);
        SDL_Rect tr = irect(top);
        SDL_RenderCopy(ren, ttop, &topsrc, &tr);
        if (bon) {
            // straight alpha: the mask in tbot's alpha removes the background
            SDL_Texture *t = premul ? tbot : tbotop;
            SDL_SetTextureBlendMode(t, premul || balpha < 1.0f ? SDL_BLENDMODE_BLEND : SDL_BLENDMODE_NONE);
            SDL_SetTextureAlphaMod(t, (Uint8)lroundf(balpha * 255));
            SDL_SetTextureScaleMode(tbotop, sm);
            SDL_Rect r = irect(br);
            SDL_RenderCopy(ren, t, nullptr, &r);
        }
        SDL_RenderPresent(ren);
        bot_rect = br; bot_on = bon; bot_hit = premul;
    }
}
#elif !defined(__EMSCRIPTEN__)
bool frontend_run() { return false; }
#endif
