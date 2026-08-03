# 5G NR PUSCH Scrambler (AVX2)

Linux user-space implementation of 5G NR uplink shared channel (PUSCH)
scrambling — **ETSI TS 38.211 v15.3.0 §6.3.1.1**, with the §5.2.1 length-31 Gold
sequence generated using Intel AVX2. C++17, no external dependencies.

## What it does

Every codeword bit is XORed with a pseudo-random sequence before transmission:

```
b~(i) = b(i) XOR c(i)          c_init = n_RNTI · 2^15 + n_ID
```

Both ends derive the same `c(n)` from the RNTI and the cell id, so the receiver
simply XORs again. The purpose is decorrelation rather than secrecy: it breaks up
long runs of identical bits, flattens the transmitted spectrum, and makes
co-scheduled users interfere as noise instead of as structured signal.

A base station repeats this for every user in every slot, and **generating `c(n)`
— not the XOR — is where the computational cost sits**. That is what this
optimises.

## Results

i5-12400F, GCC 16.1.0, `-O3 -mavx2`, best of 3000 runs. Reproduce with `make bench`.

Gold sequence generation, 1 Mbit payload:

| kernel     | time    | throughput       | vs. spec-literal |
|------------|---------|------------------|------------------|
| `scalar`   | 1866 µs | 0.54 Gbit/s      | 1.0x             |
| `wordwise` | 75.1 µs | 13.32 Gbit/s     | 24.9x            |
| `avx2`     | 39.8 µs | **25.13 Gbit/s** | **46.9x**        |

Three findings, all of which cut against the obvious expectation:

- **Most of the win is algorithmic, not SIMD.** Emitting 24 bits per LFSR
  iteration accounts for 24.9x; AVX2 adds 1.9x on top.
- **AVX2 has a fixed per-call setup cost** (eight lane start states via a GF(2)
  jump-ahead), so it only pays off above ~175 kbit — at 8460 bits it is 5.4 µs
  against the scalar kernel's 0.7 µs. AVX2 still runs unconditionally, because
  the design requires it; the cost is measured and reported rather than hidden.
- **The XOR masking barely benefits from intrinsics** (~1.0–1.2x). It is
  bandwidth-bound and the compiler already auto-vectorises it. The intrinsics
  stay because they make the vector width guaranteed, not because they are fast.

`scalar` and `wordwise` exist as measurement baselines and as correctness
oracles for the self-test. They are selectable via `--impl`, but are never a
production path — the shipped design is `avx2` at every payload size, including
the trailing partial block.

## Build

```bash
make                 # Intel oneAPI icpx by default
make CXX=g++         # or GCC / clang++
make test            # self-test + known-answer vectors
make bench           # throughput report
```

Needs C++17 and an AVX2-capable CPU (Haswell / 2013 or newer). See
[setup.md](setup.md) for oneAPI environment setup.

```bash
./pusch_scrambler scramble --in test/vectors/basic.in --out out.txt \
                           --rnti 466 --nid 270
```

Scrambling is an involution — running the output back through the same
`--rnti`/`--nid` recovers the input.

**File format.** Whitespace-separated hex bytes, `#` starts a comment. A
`# bits=N` line carries the exact bit count so lengths that are not a multiple of
8 survive a round trip. Bits are packed LSB-first: bit `i` is bit `i % 8` of byte
`i / 8`.

## Verification

Three independent sources must agree, so no single implementation is trusted:

1. **Python** — [`tools/gen_vectors.py`](tools/gen_vectors.py) transcribes the
   spec recurrences literally, sharing no code with the C++. It generates the
   `.ref` files.
2. **MATLAB 5G Toolbox** — [`matlab/verify_with_matlab.m`](matlab/verify_with_matlab.m)
   re-derives every vector with `nrPUSCHScramble`.
3. **The three C++ kernels** — cross-checked against each other.

```
$ ./pusch_scrambler selftest
PASS: 801 checks, 0 failures     # 5 RNTI × 5 n_ID × 16 lengths, all kernels

$ ./test/run_tests.sh
PASS: 54 checks                  # 9 vectors × 3 kernels × (known-answer + round-trip)
```

Vector lengths target the boundaries where this code breaks: 1 and 7 bits
(sub-byte tails), 23/24/25 (word-parallel stride), 2047/2048/2049 (AVX2 group
boundary), 8460 (realistic non-byte-aligned PUSCH `G`). Parameters cover
`c_init = 0`, max RNTI (65535) and max `n_ID` (1023).

## Design goals

| Goal | Where |
|------|-------|
| RNTI and cell id configurable at runtime | `--rnti` / `--nid`, validated 0..65535 and 0..1023 |
| Independent reference model to verify against | [`matlab/verify_with_matlab.m`](matlab/verify_with_matlab.m), [`tools/gen_vectors.py`](tools/gen_vectors.py) |
| AVX2 for the bit-wise operations | [`src/prbs.cpp`](src/prbs.cpp), [`src/scrambler.cpp`](src/scrambler.cpp) — unconditional, all sizes |
| Processing efficiency measured, not assumed | `make bench`; analysis in [DESIGN.md](DESIGN.md) |
| Test data published with the design | [`test/vectors/`](test/vectors) — 9 vectors + generator |

## Layout

```
include/                public headers
src/prbs.cpp            three Gold-sequence kernels + GF(2) jump-ahead
src/scrambler.cpp       AVX2 and scalar XOR masking
src/main.cpp            CLI: scramble / selftest / bench
matlab/                 5G Toolbox cross-check
tools/gen_vectors.py    spec-literal reference, generates test vectors
test/                   9 known-answer vectors + runner
DESIGN.md               algorithm derivation and efficiency analysis
```

## Limitations

- AVX2 is required at compile time — no runtime CPUID dispatch, no SSE fallback.
- Single-threaded; lane parallelism is within one core.
- Benchmarks come from one machine — run `make bench` on your own target.
- `matlab/verify_with_matlab.m` needs the 5G Toolbox and has not been run by the
  author on the machine these results came from. Its expected output is pinned
  regardless: `test/vectors/basic.ref` is byte-identical to output previously
  produced by `nrPUSCHScramble` for the same input and parameters.

## License

MIT — see [LICENSE](LICENSE).
