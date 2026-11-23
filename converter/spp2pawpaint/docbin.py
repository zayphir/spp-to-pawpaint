"""Reader for Substance 3D Painter's self-describing binary documents.

Files such as ``paint/document.bin`` inside an .spp start with the magic
``0x1b7c2fdd`` and hold a tree of objects. Each class is described inline
the first time it appears (class name plus a list of (field name, type
code) pairs); later objects refer to it by index. Every object is prefixed
by the absolute offset where it ends, which lets the reader verify itself.

Type codes (all little endian):
  0x01 float32            0x02/0x03/0x04 float32 vec2/vec3/vec4
  0x05 int32              0x06 int32 pair (e.g. resolutionLog2)
  0x09 int32 / enum       0x0a bool (1 byte)
  0x0b uint32             0x0c uint64 (uids, bit masks)
  0x0d float32 3x3        0x0e float32 4x4
  0x0f uint64             0x10 string (u32 length + utf-8)
  0x11 list of inline structs (u32 count, one class ref, fields)
  0x12/0x14 object        0x13 list of objects (u32 count, objects)
  0x15 uint64 (channel type flag)
  0x16 uint128 channel mask (Substance 12.2+, e.g. channelTypes; all ones = all channels)

Substance 9.1 (and probably some earlier 9.x) writes a second layout,
recognised by a 0 in the u32 after the magic (the old layout has 1 there):
  header   magic, u32 0, u32 0x48 (format version?), then the root object
           as a field value of type 0x12 at offset 12.
  object   u8 flag (0xff = null, 0 = present), u32 absolute end offset,
           string class name, u16 field count, then per field:
           string name, u8 type code, value. Class and field names are
           repeated for every object; there is no class table.
  lists    0x11 and 0x13: u32 absolute end offset, u32 count, then per
           element u8 type code + value.
  sizes    0x09 is 8 bytes (int64); 0x05 int32, 0x06 2 x int32 and 0x0b
           uint32 stay 4/8/4 bytes; other codes as above.
Both layouts produce the same dicts (field names are unchanged).
"""

import struct

MAGIC = 0x1B7C2FDD


class DocError(Exception):
    pass


class _Reader:
    def __init__(self, data):
        self.b = data
        self.classes = []
        self.errors = []
        self.unknown_types = set()

    def u32(self, o):
        return struct.unpack_from("<I", self.b, o)[0]

    def i32(self, o):
        return struct.unpack_from("<i", self.b, o)[0]

    def string(self, o):
        n = self.u32(o)
        return self.b[o + 4:o + 4 + n].decode("utf-8", "replace"), o + 4 + n

    def classref(self, o):
        ci = self.i32(o)
        o += 4
        if ci == -1:
            name, o = self.string(o)
            nf = self.u32(o)
            o += 4
            fields = []
            for _ in range(nf):
                fn, o = self.string(o)
                ft = self.u32(o)
                o += 4
                fields.append((fn, ft))
            self.classes.append((name, fields))
            ci = len(self.classes) - 1
        if ci < 0 or ci >= len(self.classes):
            raise DocError("bad class index %d at 0x%x" % (ci, o))
        return ci, o

    def obj(self, o):
        """Read one object. Damage stays local: every object knows where it
        ends, so a field this reader doesn't understand (a newer Substance
        version) skips the rest of that object only, and the object is marked
        "__partial__". Class definitions inside the skipped bytes are still
        recovered so later objects keep resolving their class indices."""
        end = self.u32(o)
        o += 4
        if end == 0:
            return None, o
        if end <= o or end > len(self.b):
            raise DocError("bad object end 0x%x at 0x%x" % (end, o))
        try:
            ci, o = self.classref(o)
        except (DocError, struct.error, UnicodeDecodeError) as e:
            self.errors.append(str(e))
            self._recover_classes(o, end)
            return {"__class": None, "__partial__": True}, end
        name, fields = self.classes[ci]
        out = {"__class": name}
        for i, (fn, ft) in enumerate(fields):
            try:
                out[fn], o = self.value(o, ft)
            except (DocError, struct.error, UnicodeDecodeError, IndexError) as e:
                self.errors.append("%s.%s: %s" % (name, fn, e))
                self._recover_classes(o, end)
                out["__partial__"] = True
                out["__unknown__"] = [f for f, _ in fields[i:]]
                return out, end
            if o > end:
                break
        if o != end:
            self.errors.append("%s ended at 0x%x, expected 0x%x" % (name, o, end))
            out["__partial__"] = True
            return out, end
        return out, o

    def _recover_classes(self, start, end):
        """Register inline class definitions found in bytes we had to skip."""
        b = self.b
        o = start
        sig = b"\xff\xff\xff\xff"
        while True:
            k = b.find(sig, o, end)
            if k < 0:
                return
            o = k + 4
            try:
                n = struct.unpack_from("<I", b, o)[0]
                if not 4 <= n <= 96:
                    continue
                name = b[o + 4:o + 4 + n]
                if not name.startswith(b"Data") or not name.isalnum():
                    continue
                p = o + 4 + n
                nf = struct.unpack_from("<I", b, p)[0]
                if nf > 256:
                    continue
                p += 4
                fields = []
                for _ in range(nf):
                    fl = struct.unpack_from("<I", b, p)[0]
                    if not 1 <= fl <= 96:
                        raise ValueError
                    fn = b[p + 4:p + 4 + fl]
                    if not fn.replace(b"_", b"").replace(b"$", b"").isalnum():
                        raise ValueError
                    ft = struct.unpack_from("<I", b, p + 4 + fl)[0]
                    if ft > 0x40:
                        raise ValueError
                    fields.append((fn.decode("ascii"), ft))
                    p += 8 + fl
                self.classes.append((name.decode("ascii"), fields))
                o = p
            except (ValueError, struct.error):
                continue

    def value(self, o, t):
        b = self.b
        if t == 0x01:
            return struct.unpack_from("<f", b, o)[0], o + 4
        if t in (0x02, 0x03, 0x04):
            return list(struct.unpack_from("<%df" % t, b, o)), o + 4 * t
        if t in (0x05, 0x09):
            return self.i32(o), o + 4
        if t == 0x06:
            return list(struct.unpack_from("<ii", b, o)), o + 8
        if t == 0x0A:
            return b[o] != 0, o + 1
        if t == 0x0B:
            return self.u32(o), o + 4
        if t in (0x0C, 0x0F, 0x15):
            return struct.unpack_from("<Q", b, o)[0], o + 8
        if t == 0x16:   # 128-bit channel mask (Substance 12.2+)
            return int.from_bytes(b[o:o + 16], "little"), o + 16
        if t == 0x0D:
            return list(struct.unpack_from("<9f", b, o)), o + 36
        if t == 0x0E:
            return list(struct.unpack_from("<16f", b, o)), o + 64
        if t == 0x10:
            return self.string(o)
        if t in (0x12, 0x14):
            return self.obj(o)
        if t == 0x13:
            n = self.u32(o)
            o += 4
            arr = []
            for _ in range(n):
                v, o = self.obj(o)
                arr.append(v)
            return arr, o
        if t == 0x11:
            n = self.u32(o)
            o += 4
            if n == 0:
                return [], o
            ci, o = self.classref(o)
            name, fields = self.classes[ci]
            arr = []
            for _ in range(n):
                el = {"__class": name}
                for fn, ft in fields:
                    el[fn], o = self.value(o, ft)
                arr.append(el)
            return arr, o
        self.unknown_types.add(t)
        raise DocError("unknown field type 0x%x at 0x%x" % (t, o))


class _ReaderV2(_Reader):
    """The self-describing layout written by Substance 9.1 (see module doc)."""

    FIXED = {0x01: "<f", 0x02: "<2f", 0x03: "<3f", 0x04: "<4f", 0x05: "<i", 0x06: "<2i",
             0x09: "<q", 0x0B: "<I", 0x0C: "<Q", 0x0D: "<9f", 0x0E: "<16f", 0x0F: "<Q", 0x15: "<Q"}

    def obj(self, o):
        flag = self.b[o]
        o += 1
        if flag == 0xFF:
            return None, o
        if flag != 0:
            raise DocError("bad object flag 0x%x at 0x%x" % (flag, o - 1))
        end = self.u32(o)
        o += 4
        if end <= o or end > len(self.b):
            raise DocError("bad object end 0x%x at 0x%x" % (end, o))
        name = "?"
        try:
            name, o = self.string(o)
            nf = struct.unpack_from("<H", self.b, o)[0]
            o += 2
            out = {"__class": name}
            for _ in range(nf):
                fn, o = self.string(o)
                t = self.b[o]
                out[fn], o = self.value(o + 1, t)
                if o > end:
                    raise DocError("ran past the object end")
        except (DocError, struct.error, UnicodeDecodeError, IndexError) as e:
            self.errors.append("%s: %s" % (name, e))
            out = locals().get("out") or {"__class": None}
            out["__partial__"] = True
            return out, end
        if o != end:
            self.errors.append("%s ended at 0x%x, expected 0x%x" % (name, o, end))
            out["__partial__"] = True
        return out, end

    def value(self, o, t):
        fmt = self.FIXED.get(t)
        if fmt is not None:
            v = struct.unpack_from(fmt, self.b, o)
            return (v[0] if len(v) == 1 else list(v)), o + struct.calcsize(fmt)
        if t == 0x0A:
            return self.b[o] != 0, o + 1
        if t == 0x16:   # 128-bit channel mask (Substance 12.2+)
            return int.from_bytes(self.b[o:o + 16], "little"), o + 16
        if t == 0x10:
            return self.string(o)
        if t in (0x12, 0x14):
            return self.obj(o)
        if t in (0x11, 0x13):
            end = self.u32(o)
            n = self.u32(o + 4)
            o += 8
            if end < o or end > len(self.b):
                raise DocError("bad list end 0x%x at 0x%x" % (end, o))
            arr = []
            try:
                for _ in range(n):
                    v, o = self.value(o + 1, self.b[o])
                    arr.append(v)
            except (DocError, struct.error, UnicodeDecodeError, IndexError) as e:
                self.errors.append("list: %s" % e)
                return arr, end          # the list knows its end: skip the rest
            if o != end:
                self.errors.append("list ended at 0x%x, expected 0x%x" % (o, end))
            return arr, end
        self.unknown_types.add(t)
        raise DocError("unknown field type 0x%x at 0x%x" % (t, o))


def layout(data):
    """1 for the classic layout, 2 for the Substance 9.1 one."""
    return 2 if struct.unpack_from("<I", data, 4)[0] == 0 else 1


def parse_with_report(data):
    """Parse a document.bin blob. Returns (root, report).

    report = {"unknown_types": [...], "errors": [...]} lists what was skipped;
    both are empty for fully understood files.
    """
    if len(data) < 20 or struct.unpack_from("<I", data, 0)[0] != MAGIC:
        raise DocError("not a Substance binary document")
    if layout(data) == 2:
        r = _ReaderV2(data)
        if data[12] not in (0x12, 0x14):
            raise DocError("unexpected root type 0x%x" % data[12])
        root, end = r.obj(13)
    else:
        r = _Reader(data)
        root, end = r.obj(20)
    if end != len(data):
        r.errors.append("document ended at 0x%x of 0x%x" % (end, len(data)))
    return root, {"unknown_types": sorted(r.unknown_types), "errors": r.errors}


def parse(data):
    """Parse a document.bin style blob into nested dicts/lists."""
    return parse_with_report(data)[0]
