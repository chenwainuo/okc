"""CPU-only affine-index lossless12/16 warp-plane packing; no escape metadata."""
import numpy as np
import torch


def indices():
    cta = np.arange(128, dtype=np.int64)[:, None, None, None, None]
    warp = np.arange(32, dtype=np.int64)[None, :, None, None, None]
    q = np.arange(4, dtype=np.int64)[None, None, :, None, None]
    lane = np.arange(32, dtype=np.int64)[None, None, None, :, None]
    v = np.arange(4, dtype=np.int64)[None, None, None, None, :]
    row = np.broadcast_to(cta * 2 + ((lane >> 2) & 1), (128, 32, 4, 32, 4))
    k = np.broadcast_to(warp * 256 + (lane >> 3) * 64 + q * 16 + (lane & 3) * 2 + (v >> 1) * 8 + (v & 1), row.shape)
    return row.reshape(4096, 4, 32, 4), k.reshape(4096, 4, 32, 4)


def encode_plane(bits, force_wide=False):
    """bits shape[4096,4,32,4], PTX B register-order halves."""
    v = np.asarray(bits, dtype=np.uint16).transpose(0, 2, 1, 3).reshape(4096, 32, 16)
    h7 = (v >> 8) & 127
    base = h7.min(axis=(1, 2)).astype(np.uint32)
    wide = h7.max(axis=(1, 2)) - base > 7
    if force_wide:
        wide[:] = True
    low = (v & 255).astype(np.uint8)
    code = np.zeros((4096, 32, 2), dtype=np.uint32)
    for q in range(4):
        for j in range(4):
            n = q * 4 + j
            offset = (h7[:, :, n].astype(np.uint32) - base[:, None]) & 7
            sign = (v[:, :, n].astype(np.uint32) >> 15) & 1
            code[:, :, q // 2] |= (offset << (8 * j + 4 * (q & 1))) | (sign << (8 * j + (3 if q & 1 else 7)))
    payload = np.zeros((4096, 1024), dtype=np.uint8)
    payload[:, :512] = low.reshape(4096, 512)
    payload[:, 512:768] = code.view(np.uint8).reshape(4096, 256)
    # Fixed direct-index wide format; same first24B/lane load positions, plus final8B/lane.
    for start, stop, off in ((0, 8, 0), (8, 12, 512), (12, 16, 768)):
        raw = np.ascontiguousarray(v[:, :, start:stop]).view(np.uint8).reshape(4096, -1)
        payload[wide, off:off + raw.shape[1]] = raw[wide]
    header = base | (wide.astype(np.uint32) << 7)
    return payload, header, {"wide": int(wide.sum()), "planes": 4096,
                             "payload_load_bytes": int(4096 * 768 + wide.sum() * 256)}


def decode_plane(payload, header):
    payload = np.asarray(payload, dtype=np.uint8)
    header = np.asarray(header, dtype=np.uint32)
    base = header & 127
    wide = (header & 128) != 0
    low = np.ascontiguousarray(payload[:, :512]).reshape(4096, 32, 16)
    code = np.ascontiguousarray(payload[:, 512:768]).view(np.uint32).reshape(4096, 32, 2)
    v = np.empty((4096, 32, 16), dtype=np.uint16)
    for q in range(4):
        for j in range(4):
            C = code[:, :, q // 2]
            high = (((C >> (8 * j + 4 * (q & 1))) & 7) + base[:, None])
            high |= ((C >> (8 * j + (3 if q & 1 else 7))) & 1) << 7
            v[:, :, 4 * q + j] = low[:, :, 4 * q + j].astype(np.uint16) | (high.astype(np.uint16) << 8)
    for start, stop, off in ((0, 8, 0), (8, 12, 512), (12, 16, 768)):
        nbytes = (stop - start) * 64
        raw = np.ascontiguousarray(payload[:, off:off + nbytes]).view(np.uint16).reshape(4096, 32, stop - start)
        v[wide, :, start:stop] = raw[wide]
    return v.reshape(4096, 32, 4, 4).transpose(0, 2, 1, 3).copy()


def pack(w):
    """Untimed, CPU, immutable weights only. Every BF16 hi/lo bit proved before GPU copy."""
    w = w.detach().cpu().contiguous()
    assert w.shape == (256, 8192) and w.dtype == torch.float32
    hi = w.to(torch.bfloat16)
    lo = (w - hi.float()).to(torch.bfloat16)
    row, k = indices()
    payloads, heads, stats = [], [], []
    for plane in (hi, lo):
        bits = plane.view(torch.int16).numpy().view(np.uint16)[row, k]
        data, head, stat = encode_plane(bits)
        assert np.array_equal(decode_plane(data, head), bits), "Lossless pack proof failed"
        payloads.append(data)
        heads.append(head)
        stats.append(stat)
    data = np.stack(payloads, axis=1).reshape(8388608)
    header = heads[0] | (heads[1] << 8)
    return torch.from_numpy(data), torch.from_numpy(header), {
        "hi": stats[0], "lo": stats[1], "allocated_bytes": data.nbytes + header.nbytes,
        "weight_transport_bytes": sum(s["payload_load_bytes"] for s in stats) + header.nbytes,
        "all_bits_exact": True, "escape_records": 0, "dependent_offset_records": 0}
