"""CPU-only layout proof for an unimplemented prepared escape-footer proposal.

This models escaped BF16 exponent XORs in the rq register layout. It performs
no CUDA/Torch work and is not a candidate or a benchmark. Overflow returns None
so a future GPU preparer must retain the current correct fallback.
"""
import json
import struct
from collections import defaultdict


FOOTER_BYTES = 128
RECORD_CAPACITY = 8


def original_patches(entries):
    """Entries are (warp, lane, q, value, seven-bit exponent_xor)."""
    patches = defaultdict(int)
    seen = set()
    for warp, lane, q, value, xor in entries:
        assert 0 <= warp < 8 and 0 <= lane < 32
        assert 0 <= q < 4 and 0 <= value < 8 and 0 < xor < 128
        coordinate = (warp, lane, q, value)
        if coordinate in seen:
            return None  # duplicate weights are not a valid escape input
        seen.add(coordinate)
        register = 4 * q + value // 2
        patches[warp, lane, register] ^= xor << (8 + 16 * (value % 2))
    return dict(patches)


def make_footer(entries):
    patches = original_patches(entries)
    if patches is None:
        return None
    lanes = defaultdict(list)
    for (warp, lane, register), xor in sorted(patches.items()):
        if xor:
            lanes[warp, lane].append(xor | register)
    if len(lanes) > RECORD_CAPACITY or any(len(words) > 2 for words in lanes.values()):
        return None
    footer = bytearray(FOOTER_BYTES)
    record = 0
    for warp in range(8):
        mask = sum(1 << lane for w, lane in lanes if w == warp)
        struct.pack_into('<I', footer, warp * 4, mask)
        struct.pack_into('<I', footer, 32 + warp * 4, record)
        for lane in range(32):
            words = lanes.get((warp, lane))
            if words:
                struct.pack_into('<II', footer, 64 + 8 * record,
                                 words[0], words[1] if len(words) == 2 else 0)
                record += 1
    return bytes(footer)


def read_footer(footer):
    patches = defaultdict(int)
    for warp in range(8):
        mask = struct.unpack_from('<I', footer, 4 * warp)[0]
        base = struct.unpack_from('<I', footer, 32 + 4 * warp)[0]
        for lane in range(32):
            if mask & (1 << lane):
                rank = (mask & ((1 << lane) - 1)).bit_count()
                word0, word1 = struct.unpack_from('<II', footer, 64 + 8 * (base + rank))
                assert word0 != 0
                for word in (word0, word1):
                    if word:
                        patches[warp, lane, word & 15] ^= word & ~15
    return dict(patches)


def stock_list_oracle(entries):
    """Independent packed-list bit path from _rqesc and the inherited PT code."""
    result = defaultdict(int)
    for warp, lane, q, value, xor in entries:
        packed = (lane << 12) | ((q * 4 + value // 2) << 8) | ((value % 2) << 7) | xor
        decoded_lane = (packed >> 12) & 31
        word = ((packed & 127) << (8 + 16 * ((packed >> 7) & 1))) | ((packed >> 8) & 15)
        result[warp, decoded_lane, word & 15] ^= word & ~15
    return dict(result)


def proof():
    singles = 0
    for warp in range(8):
        for lane in range(32):
            for q in range(4):
                for value in range(8):
                    for xor in (1, 3, 127):
                        entries = [(warp, lane, q, value, xor)]
                        assert read_footer(make_footer(entries)) == stock_list_oracle(entries)
                        singles += 1
    # Both BF16 halves of one register combine; two registers remain two words.
    mixed = [
        [(0, 31, 0, 0, 7), (0, 31, 0, 1, 31)],
        [(7, 31, 3, 6, 127), (7, 31, 0, 3, 5)],
        [(w, 31 - w, w % 4, w % 8, w + 1) for w in range(8)],
        [],
    ]
    for entries in mixed:
        assert read_footer(make_footer(entries)) == stock_list_oracle(entries)
    # A future preparer must fall back rather than truncate either overflow.
    assert make_footer([(0, lane, 0, 0, 1) for lane in range(9)]) is None
    assert make_footer([(0, 0, 0, v, 1) for v in (0, 2, 4)]) is None
    assert make_footer([(0, 0, 0, 0, 1)] * 2) is None
    # Footer is addressed by global (nb, chunk), independent of uneven K splits.
    ownership = 0
    for nb_count, chunks in ((18, 128), (32, 128), (64, 64), (64, 16)):
        for splits in (1, 2, 4, 7, 8):
            covered = []
            for split in range(splits):
                start, end = split * chunks // splits, (split + 1) * chunks // splits
                for nb in range(nb_count):
                    covered.extend(nb * chunks + ch for ch in range(start, end))
            assert sorted(covered) == list(range(nb_count * chunks))
            ownership += 1
    # 12,288-byte payload + 128-byte footer retains 128-byte stage alignment.
    assert (12288 + FOOTER_BYTES) % 128 == 0
    return {'status': 'CPU_layout_proof_pass_GPU_unimplemented',
            'single_escape_cases': singles, 'mixed_cases': len(mixed),
            'overflow_and_duplicate_fallback_cases': 3,
            'K_split_ownership_cases': ownership,
            'stage_payload_bytes': 12288, 'footer_bytes': FOOTER_BYTES,
            'payload_overhead_fraction': FOOTER_BYTES / 12288,
            'limits': 'No full codec proof, GPU preparer, kernel, compile, performance or judge evidence.'}


if __name__ == '__main__':
    print(json.dumps(proof(), indent=2))
