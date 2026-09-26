#include "common.h"

#ifdef __EMSCRIPTEN__
#include <emscripten.h>

EM_JS(void, multiplayer_init_js, (const char *room_ptr, const char *url_ptr), {
    if (globalThis.r3dsMultiplayer) return;
    const room = UTF8ToString(room_ptr) || 'world';
    const customUrl = UTF8ToString(url_ptr);
    const state = { socket: null, id: 0, connected: false, peers: [], slots: new Map(), queue: [] };
    globalThis.r3dsMultiplayer = state;
    const rebuild = (peers) => {
        state.peers = peers.filter((id) => id !== state.id);
        for (const id of [...state.slots.keys()]) if (!state.peers.includes(id)) state.slots.delete(id);
        const used = new Set(state.slots.values());
        for (const id of state.peers) {
            if (state.slots.has(id)) continue;
            const slot = [1, 2].find((n) => !used.has(n));
            if (slot === undefined) break;
            state.slots.set(id, slot);
            used.add(slot);
        }
    };
    const connect = () => {
        const scheme = location.protocol === 'https:' ? 'wss:' : 'ws:';
        const endpoint = customUrl || scheme + '/' + '/' + location.host + '/multiplayer';
        const socket = new WebSocket(endpoint + (endpoint.includes('?') ? '&' : '?') + 'room=' + encodeURIComponent(room));
        socket.binaryType = 'arraybuffer';
        state.socket = socket;
        socket.onopen = () => { state.connected = true; };
        socket.onmessage = (event) => {
            if (typeof event.data === 'string') {
                const msg = JSON.parse(event.data);
                if (msg.type === 'welcome') { state.id = msg.id; rebuild(msg.peers || []); }
                if (msg.type === 'peers') rebuild(msg.peers || []);
                return;
            }
            const data = new Uint8Array(event.data);
            const slot = state.slots.get(data[0]);
            if (slot !== undefined && data.length > 1) state.queue.push({ slot, data: data.slice(1) });
            if (state.queue.length > 120) state.queue.shift();
        };
        socket.onclose = () => {
            state.connected = false;
            state.id = 0;
            state.peers = [];
            state.slots.clear();
            state.queue.length = 0;
            setTimeout(connect, 1000);
        };
        socket.onerror = () => socket.close();
    };
    connect();
});

EM_JS(int, multiplayer_connected_js, (), {
    return globalThis.r3dsMultiplayer && globalThis.r3dsMultiplayer.connected ? 1 : 0;
});

EM_JS(int, multiplayer_peer_mask_js, (), {
    const state = globalThis.r3dsMultiplayer;
    if (!state || !state.connected) return 0;
    let mask = 1;
    for (const slot of state.slots.values()) mask |= 1 << slot;
    return mask;
});

EM_JS(int, multiplayer_send_js, (const u8 *src, int len), {
    const state = globalThis.r3dsMultiplayer;
    if (!state || !state.connected || !state.socket || state.socket.readyState !== WebSocket.OPEN) return 0;
    state.socket.send(HEAPU8.slice(src, src + len));
    return 1;
});

EM_JS(int, multiplayer_receive_js, (u8 *dst, int cap, int *slot_ptr), {
    const state = globalThis.r3dsMultiplayer;
    if (!state || !state.queue.length) return 0;
    const packet = state.queue.shift();
    const len = Math.min(cap, packet.data.length);
    HEAPU8.set(packet.data.subarray(0, len), dst);
    HEAP32[slot_ptr >> 2] = packet.slot;
    return len;
});
#endif

static bool initialized;

void multiplayer_init() {
    if (initialized) return;
    initialized = true;
#ifdef __EMSCRIPTEN__
    const char *room = getenv("R3DS_MMO_ROOM");
    const char *url = getenv("R3DS_MMO_URL");
    multiplayer_init_js(room && *room ? room : "world", url && *url ? url : "");
#endif
}

bool multiplayer_connected() {
#ifdef __EMSCRIPTEN__
    return multiplayer_connected_js() != 0;
#else
    return false;
#endif
}

u32 multiplayer_peer_mask() {
#ifdef __EMSCRIPTEN__
    return (u32)multiplayer_peer_mask_js();
#else
    return 0;
#endif
}

bool multiplayer_send(const void *data, u32 size) {
#ifdef __EMSCRIPTEN__
    return multiplayer_send_js((const u8 *)data, (int)size) != 0;
#else
    (void)data;
    (void)size;
    return false;
#endif
}

u32 multiplayer_receive(void *data, u32 capacity, u32 &slot) {
#ifdef __EMSCRIPTEN__
    int remote_slot = 0;
    int size = multiplayer_receive_js((u8 *)data, (int)capacity, &remote_slot);
    slot = (u32)remote_slot;
    return size > 0 ? (u32)size : 0;
#else
    (void)data;
    (void)capacity;
    slot = 0;
    return 0;
#endif
}
