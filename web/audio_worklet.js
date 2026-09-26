// Reads the DSP output ring that rt/web.cpp fills in shared wasm memory and
// resamples it (linear) from the 3DS rate (32728 Hz) to the device rate.
'use strict';
class FLAudio extends AudioWorkletProcessor {
  constructor() {
    super();
    this.i32 = null;
    this.port.onmessage = (e) => {
      const { mem, ring } = e.data;
      // AudioRing: i32 wr, rd, cap, rate; s16 data[2 * cap]
      this.i32 = new Int32Array(mem.buffer, ring, 4);
      this.cap = this.i32[2];
      this.s16 = new Int16Array(mem.buffer, ring + 16, 2 * this.cap);
      this.step = this.i32[3] / sampleRate;
      this.frac = 0;
      this.lastL = 0; this.lastR = 0;
    };
  }
  process(inputs, outputs) {
    const out = outputs[0];
    const L = out[0], R = out[1] || out[0];
    if (!this.i32) return true;
    const wr = Atomics.load(this.i32, 0);
    let rd = Atomics.load(this.i32, 1);
    const mask = this.cap - 1, s = this.s16;
    for (let i = 0; i < L.length; i++) {
      if (wr - rd < 2) {           // underrun: fade to silence
        this.lastL *= 0.95; this.lastR *= 0.95;
        L[i] = this.lastL; R[i] = this.lastR;
        continue;
      }
      const a = (rd & mask) * 2, b = ((rd + 1) & mask) * 2, f = this.frac;
      this.lastL = (s[a] + (s[b] - s[a]) * f) / 32768;
      this.lastR = (s[a + 1] + (s[b + 1] - s[a + 1]) * f) / 32768;
      L[i] = this.lastL; R[i] = this.lastR;
      this.frac += this.step;
      while (this.frac >= 1) { this.frac -= 1; rd++; }
    }
    // if the producer ran far ahead (tab was hidden), drop the backlog
    if (wr - rd > this.cap / 2) rd = wr - 1024;
    Atomics.store(this.i32, 1, rd);
    return true;
  }
}
registerProcessor('r3ds-audio', FLAudio);
