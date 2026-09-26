// fs:USER — RomFS served straight from the .3ds image, save/extdata/sdmc
// archives backed by host directories under save/.
#include "services.h"
#include <filesystem>
#include "platform.h"
namespace fsys = std::filesystem;

#include "rom.h"
static bool g_romfs_ready = false;
static u64 g_romfs_size = 0;
extern std::string g_save_root;
#define SAVE_ROOT g_save_root

static void romfs_init() {   // the ROM is opened by main() before the game starts
    if (g_romfs_ready) return;
    g_romfs_size = rom_romfs_size();
    g_romfs_ready = true;
}

static std::string read_path(u32 type, u32 ptr, u32 size) {
    std::string out;
    if (type == 3) {  // ascii
        for (u32 i = 0; i < size; i++) { char ch = (char)rd8(ptr + i); if (!ch) break; out += ch; }
    } else if (type == 4) {  // utf16
        for (u32 i = 0; i + 1 < size; i += 2) {
            u16 ch = rd16(ptr + i);
            if (!ch) break;
            out += ch < 0x80 ? (char)ch : '_';
        }
    } else if (type == 2) {
        char b[4];
        for (u32 i = 0; i < size; i++) { snprintf(b, sizeof b, "%02x", rd8(ptr + i)); out += b; }
    }
    return out;
}

struct FileSvc : Service {
    PFile *fd = nullptr;
    bool rom = false;
    std::string host;
    FileSvc(std::string n) : Service(n) {}
    ~FileSvc() { if (fd && !rom) pf_close(fd); }
    u64 size() {
        if (rom) return g_romfs_size;
        return pf_size(fd);
    }
    void handle(Ipc &ipc) override {
        switch (ipc.cmd()) {
        case 0x0802: {  // Read(off64, size, desc, ptr)
            u64 off = ipc.p(1) | ((u64)ipc.p(2) << 32);
            u32 sz = ipc.p(3), dst = ipc.p(5);
            s64 n = 0;
            if (rom) {
                n = rom_romfs_read(gp(dst), sz, off);
            } else {
                n = pf_pread(fd, gp(dst), sz, off);
            }
            if (n < 0) n = 0;
            DBG("[fs] read %s off=%llx size=%x -> %lld", name.c_str(), (unsigned long long)off, sz, (long long)n);
            ipc.reply(2, 2); ipc.w(2, (u32)n); ipc.w(3, (sz << 4) | 0xC); ipc.w(4, dst);
            break;
        }
        case 0x0803: {  // Write(off64, size, flags, desc, ptr)
            u64 off = ipc.p(1) | ((u64)ipc.p(2) << 32);
            u32 sz = ipc.p(3), src = ipc.p(6);
            s64 n = rom ? 0 : pf_pwrite(fd, gp(src), sz, off);
            if (n < 0) n = 0;
            ipc.reply(2, 2); ipc.w(2, (u32)n); ipc.w(3, (sz << 4) | 0xA); ipc.w(4, src);
            break;
        }
        case 0x0804: { u64 s = size(); ipc.reply(3, 0); ipc.w(2, (u32)s); ipc.w(3, (u32)(s >> 32)); break; }
        case 0x0805: { if (!rom) pf_truncate(fd, ipc.p(1) | ((u64)ipc.p(2) << 32)); ipc.reply(1, 0); break; }
        case 0x0808: ipc.reply(1, 0); break;  // Close
        case 0x0809: ipc.reply(1, 0); break;  // Flush
        case 0x080A: case 0x080B: ipc.reply(1, 0); break;  // SetPriority etc
        case 0x080C: {  // OpenLinkFile
            auto f = std::make_shared<FileSvc>(name);
            f->rom = rom; f->host = host;
            if (!rom) f->fd = pf_open(host.c_str(), true, false);
            ipc.reply(1, 2); ipc.w(2, 0x10); ipc.w(3, new_session_handle(f));
            break;
        }
        default: unknown(ipc);
        }
    }
};

struct DirSvc : Service {
    std::vector<fsys::directory_entry> ents;
    size_t pos = 0;
    DirSvc(std::string n) : Service(n) {}
    void handle(Ipc &ipc) override {
        switch (ipc.cmd()) {
        case 0x0801: {  // Read(count, desc, ptr)
            u32 cnt = ipc.p(1), dst = ipc.p(3), n = 0;
            for (; n < cnt && pos < ents.size(); n++, pos++) {
                u32 e = dst + n * 0x228;
                memset(gp(e), 0, 0x228);
                std::string nm = ents[pos].path().filename().string();
                for (size_t i = 0; i < nm.size() && i < 0x105; i++) wr16(e + i * 2, (u8)nm[i]);
                bool dir = ents[pos].is_directory();
                wr8(e + 0x21C, dir); wr8(e + 0x21E, 0);
                wr64(e + 0x220, dir ? 0 : (u64)ents[pos].file_size());
            }
            ipc.reply(2, 2); ipc.w(2, n); ipc.w(3, (cnt * 0x228 << 4) | 0xC); ipc.w(4, dst);
            break;
        }
        case 0x0802: ipc.reply(1, 0); break;
        default: unknown(ipc);
        }
    }
};

struct Archive { u32 id; std::string root; bool rom; };
static std::map<u64, Archive> g_archives;
static u64 g_arch_next = 0x100000001ull;

static std::string archive_root(u32 id, u32 ptype, u32 pptr, u32 psize) {
    switch (id) {
    case 4: return SAVE_ROOT + "/savedata";
    case 6: case 7: return SAVE_ROOT + "/extdata/" + read_path(2, pptr, psize);
    case 9: return SAVE_ROOT + "/sdmc";
    default: return SAVE_ROOT + "/archive_" + std::to_string(id);
    }
}

static Result open_archive(u32 id, u32 ptype, u32 pptr, u32 psize, u64 &out) {
    Archive a{id, "", id == 3};
    if (id == 3 || id == 0x2345678A) { romfs_init(); a.rom = true; }
    else {
        a.root = archive_root(id, ptype, pptr, psize);
        if (id == 4 && !fsys::exists(a.root)) {
            INFO("[fs] savedata not formatted yet");
            return 0xC8A04554;
        }
        // extdata / shared extdata: a real console ships the OS-owned archives
        // already created, and games also open their own before creating it —
        // provision an empty archive either way. If the game needed actual
        // content, the file-open failure deeper down names what's missing.
        if ((id == 6 || id == 7) && !fsys::exists(a.root))
            INFO("[fs] extdata %s missing — auto-creating it empty", a.root.c_str());
        fsys::create_directories(a.root);
    }
    out = g_arch_next++;
    g_archives[out] = a;
    INFO("[fs] OpenArchive id=%x path(%u)=%s -> %llx", id, ptype, read_path(ptype, pptr, psize).c_str(), (unsigned long long)out);
    return RES_OK;
}

static Result open_file(const Archive &a, u32 ptype, u32 pptr, u32 psize, u32 flags, u32 &handle) {
    auto f = std::make_shared<FileSvc>("file");
    if (a.rom) {
        romfs_init();
        f->rom = true;
        f->name = "romfs";
    } else {
        std::string p = read_path(ptype, pptr, psize);
        f->host = a.root + p;
        f->name = f->host;
        f->fd = pf_open(f->host.c_str(), (flags & 2) != 0, (flags & 4) != 0);
        if (!f->fd) { INFO("[fs] open %s -> not found", f->host.c_str()); return 0xC8804478; }
    }
    handle = new_session_handle(f);
    INFO("[fs] OpenFile %s flags=%x -> %08x", f->name.c_str(), flags, handle);
    return RES_OK;
}

struct FsService : Service {
    FsService(std::string n) : Service(n) {}
    void handle(Ipc &ipc) override {
        switch (ipc.cmd()) {
        case 0x0801: ipc.reply(1, 0); break;  // Initialize(desc, program info buf): nothing to retain
        case 0x0802: {  // OpenFile(trans, arch64, ptype, psize, flags, attr, desc, ptr)
            u64 ah = ipc.p(2) | ((u64)ipc.p(3) << 32);
            auto it = g_archives.find(ah);
            u32 h = 0;
            Result r = it == g_archives.end() ? 0xC8804470 : open_file(it->second, ipc.p(4), ipc.p(9), ipc.p(5), ipc.p(6), h);
            ipc.reply(1, 2, r); ipc.w(2, 0x10); ipc.w(3, h);
            break;
        }
        case 0x0803: {  // OpenFileDirectly
            u32 aid = ipc.p(2), aptype = ipc.p(3), apsize = ipc.p(4), fptype = ipc.p(5), fpsize = ipc.p(6), flags = ipc.p(7);
            u32 aptr = ipc.p(10), fptr = ipc.p(12);
            u64 ah;
            u32 h = 0;
            Result r = open_archive(aid, aptype, aptr, apsize, ah);
            if (r == RES_OK) r = open_file(g_archives[ah], fptype, fptr, fpsize, flags, h);
            ipc.reply(1, 2, r); ipc.w(2, 0x10); ipc.w(3, h);
            break;
        }
        case 0x0804: {  // DeleteFile(trans, arch64, ptype, psize, desc, ptr)
            u64 ah = ipc.p(2) | ((u64)ipc.p(3) << 32);
            std::string p = g_archives[ah].root + read_path(ipc.p(4), ipc.p(7), ipc.p(5));
            ipc.reply(1, 0, fsys::remove(p) ? RES_OK : 0xC8804478);
            break;
        }
        case 0x0806: case 0x0807: {  // DeleteDirectory(Recursively)
            u64 ah = ipc.p(2) | ((u64)ipc.p(3) << 32);
            std::string p = g_archives[ah].root + read_path(ipc.p(4), ipc.p(7), ipc.p(5));
            fsys::remove_all(p);
            ipc.reply(1, 0);
            break;
        }
        case 0x0808: {  // CreateFile(trans, arch64, ptype, psize, attr, size64, desc, ptr)
            u64 ah = ipc.p(2) | ((u64)ipc.p(3) << 32);
            std::string p = g_archives[ah].root + read_path(ipc.p(4), ipc.p(10), ipc.p(5));
            u64 sz = ipc.p(7) | ((u64)ipc.p(8) << 32);
            Result r = RES_OK;
            if (fsys::exists(p)) r = 0xC82044BE;
            else { PFile *nf = pf_open(p.c_str(), true, true); if (nf) { pf_truncate(nf, sz); pf_close(nf); } }
            INFO("[fs] CreateFile %s size=%llx -> %08x", p.c_str(), (unsigned long long)sz, r);
            ipc.reply(1, 0, r);
            break;
        }
        case 0x0809: {  // CreateDirectory(trans, arch64, ptype, psize, attr, desc, ptr)
            u64 ah = ipc.p(2) | ((u64)ipc.p(3) << 32);
            std::string p = g_archives[ah].root + read_path(ipc.p(4), ipc.p(8), ipc.p(5));
            Result r = fsys::exists(p) ? 0xC82044B9 : RES_OK;
            fsys::create_directories(p);
            ipc.reply(1, 0, r);
            break;
        }
        case 0x0805: case 0x080A: {  // RenameFile / RenameDirectory
            LOG("[fs] rename not implemented");
            ipc.reply(1, 0); break;
        }
        case 0x080B: {  // OpenDirectory(arch64, ptype, psize, desc, ptr)
            u64 ah = ipc.p(1) | ((u64)ipc.p(2) << 32);
            std::string p = g_archives[ah].root + read_path(ipc.p(3), ipc.p(6), ipc.p(4));
            if (!fsys::is_directory(p)) { ipc.reply(1, 2, 0xC8804478); ipc.w(2, 0x10); ipc.w(3, 0); break; }
            auto d = std::make_shared<DirSvc>(p);
            for (auto &e : fsys::directory_iterator(p)) d->ents.push_back(e);
            ipc.reply(1, 2); ipc.w(2, 0x10); ipc.w(3, new_session_handle(d));
            break;
        }
        case 0x080C: {  // OpenArchive(id, ptype, psize, desc, ptr)
            u64 ah = 0;
            Result r = open_archive(ipc.p(1), ipc.p(2), ipc.p(5), ipc.p(3), ah);
            ipc.reply(3, 0, r); ipc.w(2, (u32)ah); ipc.w(3, (u32)(ah >> 32));
            break;
        }
        case 0x080E: ipc.reply(1, 0); break;  // CloseArchive
        case 0x0812: {  // GetFreeBytes
            ipc.reply(3, 0); ipc.w(2, 0x10000000); ipc.w(3, 0); break;
        }
        case 0x0814: case 0x0817: case 0x0818: case 0x0821:  // sdmc/card queries
            ipc.reply(2, 0); ipc.w(2, 1); break;
        case 0x0845: {  // GetFormatInfo(archive id, ptype, psize, desc, ptr)
            u32 id = ipc.p(1);
            std::string root = archive_root(id, ipc.p(2), ipc.p(5), ipc.p(3));
            bool ok = fsys::exists(root);
            ipc.reply(5, 0, ok ? RES_OK : 0xC8A04554); ipc.w(2, 0x100000); ipc.w(3, 32); ipc.w(4, 64); ipc.w(5, 0);
            break;
        }
        case 0x084C: {  // FormatSaveData(archive id, ptype, psize, blocks, dirs, files, dirbuckets, filebuckets, dupdata, desc, ptr)
            fsys::remove_all(SAVE_ROOT + "/savedata");
            fsys::create_directories(SAVE_ROOT + "/savedata");
            INFO("[fs] FormatSaveData");
            ipc.reply(1, 0);
            break;
        }
        case 0x0851: {  // CreateExtSaveData(info[4], ...)
            char b[64];
            snprintf(b, sizeof b, "%08x%08x%08x", ipc.p(1), ipc.p(2), ipc.p(3));
            u32 info = ipc.p(1);
            (void)info;
            // path for OpenArchive is the binary {media, lo, hi}
            u8 raw[12]; memcpy(raw, gp(ipc.buf + 4), 12);
            std::string key; char h[4];
            for (int i = 0; i < 12; i++) { snprintf(h, sizeof h, "%02x", raw[i]); key += h; }
            fsys::create_directories(SAVE_ROOT + "/extdata/" + key);
            INFO("[fs] CreateExtSaveData %s", key.c_str());
            ipc.reply(1, 0);
            break;
        }
        case 0x0852: ipc.reply(1, 0); break;  // DeleteExtSaveData
        case 0x0861: case 0x0862: case 0x0856:  // InitializeWithSdkVersion / SetPriority / ControlArchive?
            ipc.reply(1, 0); break;
        case 0x0863: ipc.reply(2, 0); ipc.w(2, 0); break;  // GetPriority
        case 0x0849: case 0x0854: {  // GetArchiveResource / GetSdmcArchiveResource
            ipc.reply(5, 0); ipc.w(2, 0x200); ipc.w(3, 0x8000); ipc.w(4, 0x80000); ipc.w(5, 0x80000); break;
        }
        case 0x0862 + 0x100: default: unknown(ipc);
        }
    }
};

void register_fs() {
    services_register("fs:USER", [] { return std::make_shared<FsService>("fs:USER"); });
}
