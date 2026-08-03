# Design notes

## 1. The specification

**§6.3.1.1** — the codeword is masked bit by bit:

```
b~(i) = ( b(i) + c(i) ) mod 2          c_init = n_RNTI · 2^15 + n_ID
```

`n_ID` is `dataScramblingIdentityPUSCH` (0..1023) when configured, otherwise the
cell id (0..1007). The CLI accepts the wider range.

**§5.2.1** — `c(n)` is a length-31 Gold sequence, first `Nc = 1600` outputs
discarded:

```
c(n)     = ( x1(n + Nc) + x2(n + Nc) ) mod 2
x1(n+31) = ( x1(n+3) + x1(n) ) mod 2
x2(n+31) = ( x2(n+3) + x2(n+2) + x2(n+1) + x2(n) ) mod 2
x1(0) = 1, x1(1..30) = 0        x2(i) = (c_init >> i) & 1
```

The masking XOR is trivially parallel. All the engineering is in generating `c(n)`.

## 2. State layout

Each LFSR state is a 31-bit register where **bit `i` holds `x(n + i)`**. One step
is a right shift with feedback entering at bit 30. The output bit is always
`s & 1`, and the taps sit at *low* bit positions — which is what makes everything
below possible.

## 3. `scalar` — literal transcription

One bit per iteration, straight from the spec. Kept as the correctness oracle and
the benchmark baseline. **0.54 Gbit/s** — a single PUSCH carrier needs 1–2 Gbit/s,
so the obvious implementation is already too slow.

## 4. `wordwise` — 24 bits per iteration

The key observation: **the lowest tap is at offset 3**, so the next 28 output bits
are already determined by bits in the register. 24 are taken per iteration rather
than 28 because that is byte-aligned (3 bytes out, no bit shuffling).

```c
nb_x1 = ((s >> 3) ^ s) & 0xFFFFFF;      // x(n+31) .. x(n+54)
s     = (s >> 24) | (nb << 7);          // now holds x(n+24) .. x(n+54)
```

The `x2` tap polynomial factors, halving its shift count:

```c
t  = s ^ (s >> 1);        // x(n)   + x(n+1)
nb = t ^ (t >> 2);        // + x(n+2) + x(n+3)   -> 2 shifts, 2 XORs, not 4+3
```

**13.32 Gbit/s — 24.9x, with no SIMD at all.** The single largest win in the
project, and it comes from the recurrence, not the instruction set.

## 5. `avx2` — eight lanes

Eight lanes each generate one *contiguous* chunk of the output, which raises two
problems.

**Where does each lane start?** The recurrences are **linear over GF(2)**, so
advancing `k` steps is multiplication by a fixed 31×31 matrix `M^k`, computed by
repeated squaring in `O(log k)`. The same machinery collapses the mandatory
`Nc = 1600` advance into one matrix application. (`x1`'s initial state is fixed by
the spec, so its post-`Nc` value does not depend on `c_init` at all.)

**How do eight lanes reach memory?** AVX2 has no scatter. Each lane emits a
32-bit word per iteration, built from two 16-bit substeps — a 31-bit register
cannot emit 32 bits at once:

```c
emit = (x1 ^ x2) & 0xFFFF;
x    = (x >> 16) | (nb << 15);          // 15 old + 16 new = 31 bits
```

Eight iterations are accumulated into an 8×8 word matrix and transposed with
`VPUNPCKLDQ` / `VPUNPCKHDQ` / `VPERM2I128`; each register then holds eight
consecutive words of one chunk and is written with a single 32-byte store. The
transpose costs ~24 instructions per 2048 bits generated.

Every operation is a genuine AVX2 bit-wise instruction — `VPXOR`, `VPAND`,
`VPOR`, `VPSRLD`, `VPSLLD` — checked against Intel's published intrinsics data
(guide v3.6.9).

The trailing partial group generates one complete 2048-bit vector group and keeps
what it needs, discarding at most 2047 bits. That is cheaper than a scalar tail
and keeps every generated bit on the AVX2 path.

## 6. Setup cost, measured

The jump-ahead is `O(log k)` but not free, and runs **once per call**:

| payload    | `wordwise` | `avx2`  | faster   |
|------------|------------|---------|----------|
| 8 460 bit  | 0.7 µs     | 5.4 µs  | wordwise |
| 50 000     | 3.7 µs     | 7.4 µs  | wordwise |
| 80 000     | 6.2 µs     | 9.8 µs  | wordwise |
| 150 000    | 11.1 µs    | 11.6 µs | wordwise |
| **200 000**| 14.8 µs    | 13.6 µs | **avx2** |
| 500 000    | 37.8 µs    | 23.4 µs | avx2     |
| 1 000 000  | 75.1 µs    | 39.8 µs | avx2     |

Crossover ≈ **175 kbit**. Realistic PUSCH codewords are 8 k–300 kbit, so much of
real traffic sits on the wrong side of it.

**Reported, not routed around.** The design requires AVX2 for the bit-wise
operations, so AVX2 runs unconditionally; quantifying what that costs is exactly
what the efficiency requirement asks for. Substituting a scalar path below the
crossover would produce a better benchmark and a different answer than the one
asked for.

Two notes on reducing it further:

- Reusing lane 7's final state for the tail, instead of recomputing a matrix
  power, already removed two of four `mat_pow` calls (11.1 → 5.4 µs at 8460 bits).
- What remains is dominated by `mat_pow`. Replacing the 31×31 matrix with
  polynomial arithmetic in `GF(2)[x]/p(x)` — advancing `k` steps is `x^k mod p`,
  a 31-bit value — would shrink it and move the crossover down.

## 7. Efficiency

1 Mbit payload, i5-12400F, GCC 16.1.0, `-O3 -mavx2`, best-of:

| stage             | time    | Gbit/s | tsc/bit |
|-------------------|---------|--------|---------|
| PRBS `scalar`     | 1866 µs | 0.54   | 4.66    |
| PRBS `wordwise`   | 75.1 µs | 13.32  | 0.188   |
| PRBS `avx2`       | 39.8 µs | 25.13  | 0.099   |
| XOR mask `scalar` | 3.2 µs  | 312    | 0.008   |
| XOR mask `avx2`   | 2.6 µs  | 385    | 0.007   |
| end-to-end        | 44.3 µs | 22.57  | 0.111   |

- Generation dominates; masking is ~6% of end-to-end time.
- AVX2 contributes 1.9x; the algorithm contributes 24.9x. Reporting only the
  combined 47x would misattribute the result.
- The XOR gains 1.0–1.2x from intrinsics. At 125 kB the buffers are L2-resident
  and the loop is bandwidth-bound; GCC also auto-vectorises the scalar version.
  At 100 Mbit it drops to ~90 Gbit/s, confirming the bound.

**On measuring cycles.** `rdtsc` counts the invariant TSC, which ticks at base
frequency and does not track turbo. The benchmark therefore reports wall-clock
ns and Gbit/s as the primary figures, labels the TSC column `tsc/bit` rather than
"cycles/bit", and prints the measured TSC rate. Dividing TSC ticks by elapsed
time recovers the base clock and nothing more. Each figure is the best of N runs
after a warm-up — minimum is the right statistic for throughput microbenchmarks,
being least disturbed by interrupts and scheduling.

## 8. Verification and portability

`.ref` files come from a slow literal transcription of §5.2.1 in Python that
shares no code with the C++; MATLAB's `nrPUSCHScramble` independently reproduces
them; the three kernels are cross-checked against each other over 5 RNTI × 5
`n_ID` × 16 lengths, and every vector is round-tripped to confirm the involution.

Target is Linux user space on AVX2-capable x86-64 (Haswell / 2013 or newer). AVX2
is assumed at compile time — no runtime CPUID dispatch, no SSE fallback,
single-threaded. The only compiler-specific constructs are `<immintrin.h>`
intrinsics and `__rdtsc()` / `_mm_lfence()` in the benchmark, which GCC, Clang
and icpx all provide identically.
