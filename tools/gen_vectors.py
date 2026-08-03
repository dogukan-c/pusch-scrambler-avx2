#!/usr/bin/env python3
"""Generate known-answer test vectors for the PUSCH scrambler.

The reference sequence here is a literal transcription of the recurrences in
ETSI TS 38.211 v15.3.0 Section 5.2.1, deliberately written the slow, obvious
way. It shares no code or optimisation with the C++ implementation, so a match
is independent evidence rather than a tautology.

Usage: python3 tools/gen_vectors.py [output-dir]
"""

import os
import random
import sys

NC = 1600
MASK31 = (1 << 31) - 1


def gold(n_bits, cinit):
    """c(n) for n in [0, n_bits), TS 38.211 5.2.1."""
    x1 = [0] * (NC + n_bits + 31)
    x2 = [0] * (NC + n_bits + 31)
    x1[0] = 1
    for i in range(31):
        x2[i] = (cinit >> i) & 1
    for n in range(NC + n_bits):
        x1[n + 31] = x1[n + 3] ^ x1[n]
        x2[n + 31] = x2[n + 3] ^ x2[n + 2] ^ x2[n + 1] ^ x2[n]
    return [x1[n + NC] ^ x2[n + NC] for n in range(n_bits)]


def cinit_pusch(rnti, nid):
    """TS 38.211 6.3.1.1."""
    return ((rnti << 15) + nid) & MASK31


def pack(bits):
    """LSB-first packing: bit i -> byte i//8, position i%8."""
    out = bytearray((len(bits) + 7) // 8)
    for i, b in enumerate(bits):
        if b:
            out[i >> 3] |= 1 << (i & 7)
    return bytes(out)


def unpack(data, n_bits):
    return [(data[i >> 3] >> (i & 7)) & 1 for i in range(n_bits)]


def read_payload(path):
    """Reads an existing .in file so committed vectors stay byte-stable."""
    data = bytearray()
    with open(path) as f:
        for line in f:
            line = line.split("#", 1)[0]
            data += bytes(int(tok, 16) for tok in line.split())
    return bytes(data)


def write(path, data, n_bits, header):
    with open(path, "w", newline="\n") as f:
        f.write(f"# {header}\n")
        f.write(f"# bits={n_bits}\n")
        for i in range(0, len(data), 16):
            f.write(" ".join(f"{b:02X}" for b in data[i:i + 16]) + "\n")


# name, rnti, nid, bits, why this case exists
CASES = [
    ("basic",         466,   270,  80000, "10000-byte reference payload"),
    ("zero_params",     0,     0,    512, "c_init = 0, the degenerate seed"),
    ("max_rnti",    65535,  1007,   4096, "max n_RNTI with max cell id"),
    ("max_nid",         1,  1023,   2048, "max dataScramblingIdentityPUSCH; exactly one AVX2 group"),
    ("group_edge",    100,   300,   2049, "one bit past an AVX2 group, forces the scalar tail"),
    ("under_group",  1234,   555,   2047, "one bit short of a group, pure tail path"),
    ("unaligned",     466,   270,   8460, "realistic PUSCH G, not a multiple of 8"),
    ("tiny",           17,    42,      1, "single bit"),
    ("byte_edge",      99,   777,      7, "sub-byte length, tail masking"),
]


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "test/vectors"
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(0xC0FFEE)  # fixed seed: vectors are reproducible

    for name, rnti, nid, bits, why in CASES:
        in_path = os.path.join(out_dir, name + ".in")
        nbytes = (bits + 7) // 8
        if os.path.exists(in_path):
            payload = read_payload(in_path)
            if len(payload) != nbytes:
                raise SystemExit(f"{in_path}: has {len(payload)} bytes, expected {nbytes}")
        else:
            payload = bytearray(rng.getrandbits(8) for _ in range(nbytes))
            if bits % 8:
                payload[-1] &= (1 << (bits % 8)) - 1
            payload = bytes(payload)

        c = gold(bits, cinit_pusch(rnti, nid))
        scrambled = pack([a ^ b for a, b in zip(unpack(payload, bits), c)])

        write(in_path, payload, bits, f"rnti={rnti} nid={nid} :: {why}")
        write(os.path.join(out_dir, name + ".ref"), scrambled, bits,
              f"expected output for rnti={rnti} nid={nid}")
        print(f"{name:<13} rnti={rnti:<6} nid={nid:<5} bits={bits:<7} {why}")

    print(f"\n{len(CASES)} vectors written to {out_dir}/")


if __name__ == "__main__":
    main()
