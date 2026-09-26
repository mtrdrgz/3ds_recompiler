#pragma once
#include "common.h"
struct InputState { u32 buttons; s16 cx, cy; u16 tx, ty; u32 touch; };
InputState input_get();
void input_set(const InputState &s);
// 3DS pad bits
enum : u32 { BTN_A = 1, BTN_B = 2, BTN_SELECT = 4, BTN_START = 8, BTN_DRIGHT = 16, BTN_DLEFT = 32, BTN_DUP = 64,
             BTN_DDOWN = 128, BTN_R = 256, BTN_L = 512, BTN_X = 1024, BTN_Y = 2048 };
