"""Minimal reader for Qt QDataStream QVariantMap blobs (big endian).

Substance uses these for small metadata headers, e.g. on texture-cache
entries and .image resources.
"""

import struct


def _qstring(b, o):
    n = struct.unpack_from(">I", b, o)[0]
    o += 4
    if n == 0xFFFFFFFF:
        return None, o
    return b[o:o + n].decode("utf-16-be"), o + n


def _qvariant(b, o):
    t = struct.unpack_from(">I", b, o)[0]
    o += 5  # type + is-null byte
    if t == 1:
        return b[o] != 0, o + 1
    if t == 2:
        return struct.unpack_from(">i", b, o)[0], o + 4
    if t == 3:
        return struct.unpack_from(">I", b, o)[0], o + 4
    if t == 4:
        return struct.unpack_from(">q", b, o)[0], o + 8
    if t == 5:
        return struct.unpack_from(">Q", b, o)[0], o + 8
    if t == 6:
        return struct.unpack_from(">d", b, o)[0], o + 8
    if t == 8:
        return read_map(b, o)
    if t == 10:
        return _qstring(b, o)
    if t == 12:
        n = struct.unpack_from(">I", b, o)[0]
        o += 4
        if n == 0xFFFFFFFF:
            return b"", o
        return b[o:o + n], o + n
    raise ValueError("unsupported QVariant type %d at %d" % (t, o))


def read_map(b, o=0):
    """Return (dict, offset after the map)."""
    n = struct.unpack_from(">I", b, o)[0]
    o += 4
    d = {}
    for _ in range(n):
        k, o = _qstring(b, o)
        v, o = _qvariant(b, o)
        d[k] = v
    return d, o
