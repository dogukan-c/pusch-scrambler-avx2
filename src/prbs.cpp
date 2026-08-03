#include "prbs.hpp"

#include <immintrin.h>

#include <cstring>

namespace pusch {
namespace {

constexpr uint32_t kMask31 = 0x7FFFFFFFu;
constexpr uint32_t kTapX1 = 0x9u;  // x1(n+31) = x1(n+3) + x1(n)
constexpr uint32_t kTapX2 = 0xFu;  // x2(n+31) = x2(n+3) + x2(n+2) + x2(n+1) + x2(n)
constexpr unsigned kNc = 1600;     // mandatory init offset, TS 38.211 5.2.1

// State layout: a 31-bit register whose bit i holds x(n + i). Advancing the
// LFSR by one is a right shift with the feedback bit entering at bit 30.

// ---------------------------------------------------------------------------
// GF(2) state-transition matrix.
//
// The recurrences are linear over GF(2), so advancing k steps is multiplication
// by a fixed 31x31 matrix M^k. Row i is the mask of old-state bits that XOR into
// new bit i. This turns both the Nc=1600 init advance and the per-lane offsets
// used by the AVX2 kernel into O(log k) work instead of O(k).
//
// Setup path only - never runs inside a generation loop.
// ---------------------------------------------------------------------------
struct Mat {
    uint32_t row[31];
};

unsigned parity32(uint32_t v) {
    v ^= v >> 16;
    v ^= v >> 8;
    v ^= v >> 4;
    v ^= v >> 2;
    v ^= v >> 1;
    return v & 1u;
}

Mat step_matrix(uint32_t tap) {
    Mat m{};
    for (unsigned i = 0; i < 30; ++i) m.row[i] = 1u << (i + 1);
    m.row[30] = tap;
    return m;
}

uint32_t mat_apply(const Mat& m, uint32_t s) {
    uint32_t r = 0;
    for (unsigned i = 0; i < 31; ++i) r |= parity32(m.row[i] & s) << i;
    return r;
}

Mat mat_mul(const Mat& a, const Mat& b) {
    Mat c{};
    for (unsigned i = 0; i < 31; ++i) {
        uint32_t r = 0;
        for (unsigned j = 0; j < 31; ++j)
            if ((a.row[i] >> j) & 1u) r ^= b.row[j];
        c.row[i] = r;
    }
    return c;
}

Mat mat_pow(Mat m, uint64_t k) {
    Mat r{};
    for (unsigned i = 0; i < 31; ++i) r.row[i] = 1u << i;  // identity
    while (k) {
        if (k & 1) r = mat_mul(r, m);
        m = mat_mul(m, m);
        k >>= 1;
    }
    return r;
}

struct Seed {
    uint32_t x1, x2;
};

// x1's initial state is fixed by the spec (x1(0)=1, x1(1..30)=0), so its
// post-Nc value does not depend on c_init at all.
Seed seed_after_nc(uint32_t cinit) {
    static const Mat p1 = mat_pow(step_matrix(kTapX1), kNc);
    static const Mat p2 = mat_pow(step_matrix(kTapX2), kNc);
    return {mat_apply(p1, 1u), mat_apply(p2, cinit & kMask31)};
}

// ---------------------------------------------------------------------------
// Shared bit recurrences.
//
// Because the lowest tap sits at offset 3, the next 28 output bits of a
// register depend only on bits already present, so up to 28 steps can be
// produced with a handful of shifts. next_x1/next_x2 return k fresh bits
// (k <= 28, selected by mask); advance() folds them back into the state.
// ---------------------------------------------------------------------------
inline uint32_t next_x1(uint32_t s, uint32_t mask) {
    return ((s >> 3) ^ s) & mask;
}

inline uint32_t next_x2(uint32_t s, uint32_t mask) {
    const uint32_t t = s ^ (s >> 1);  // x(n) + x(n+1)
    return (t ^ (t >> 2)) & mask;     // + x(n+2) + x(n+3), two ops instead of four
}

inline uint32_t advance(uint32_t s, uint32_t nb, unsigned k) {
    return ((s >> k) | (nb << (31 - k))) & kMask31;
}

// --------------------------------------------------------------- 1. scalar
void prbs_scalar(uint8_t* out, size_t num_bits, const Seed& sd) {
    uint32_t x1 = sd.x1, x2 = sd.x2;
    for (size_t i = 0; i < num_bits; ++i) {
        out[i >> 3] |= static_cast<uint8_t>(((x1 ^ x2) & 1u) << (i & 7));
        x1 = advance(x1, next_x1(x1, 1u), 1);
        x2 = advance(x2, next_x2(x2, 1u), 1);
    }
}

// ------------------------------------------------------------- 2. wordwise
// 24 bits (3 bytes) per iteration. Caller must start on a byte boundary.
void prbs_wordwise_from(uint8_t* out, size_t num_bits, uint32_t x1, uint32_t x2) {
    constexpr uint32_t kM24 = 0xFFFFFFu;
    size_t done = 0;
    while (done + 24 <= num_bits) {
        const uint32_t w = (x1 ^ x2) & kM24;
        uint8_t* p = out + (done >> 3);
        p[0] = static_cast<uint8_t>(w);
        p[1] = static_cast<uint8_t>(w >> 8);
        p[2] = static_cast<uint8_t>(w >> 16);
        x1 = advance(x1, next_x1(x1, kM24), 24);
        x2 = advance(x2, next_x2(x2, kM24), 24);
        done += 24;
    }
    if (done < num_bits) {
        const uint32_t w = (x1 ^ x2) & kM24;
        for (size_t i = done; i < num_bits; ++i)
            out[i >> 3] |= static_cast<uint8_t>(((w >> (i - done)) & 1u) << (i & 7));
    }
}

// ----------------------------------------------------------------- 3. AVX2
// Eight independent lanes, each generating one contiguous chunk of the output.
// Per iteration a lane emits 32 bits as two 16-bit substeps; eight iterations
// fill an 8x8 word matrix which is transposed so every lane's eight words land
// contiguously and can be written with a single 32-byte store.
constexpr size_t kLaneGroupBits = 256;   // 8 words x 32 bits, per lane per group
constexpr size_t kGroupBits = 8 * kLaneGroupBits;

inline __m256i v_next_x1(__m256i s, __m256i mask) {
    return _mm256_and_si256(_mm256_xor_si256(_mm256_srli_epi32(s, 3), s), mask);
}

inline __m256i v_next_x2(__m256i s, __m256i mask) {
    const __m256i t = _mm256_xor_si256(s, _mm256_srli_epi32(s, 1));
    return _mm256_and_si256(_mm256_xor_si256(t, _mm256_srli_epi32(t, 2)), mask);
}

// Emits the low 16 output bits of every lane and advances both registers by 16.
inline __m256i v_step16(__m256i& x1, __m256i& x2, __m256i mask) {
    const __m256i emit = _mm256_and_si256(_mm256_xor_si256(x1, x2), mask);
    const __m256i n1 = v_next_x1(x1, mask);
    const __m256i n2 = v_next_x2(x2, mask);
    // new state = (s >> 16) | (fresh << 15): 15 old bits + 16 new = 31 bits.
    x1 = _mm256_or_si256(_mm256_srli_epi32(x1, 16), _mm256_slli_epi32(n1, 15));
    x2 = _mm256_or_si256(_mm256_srli_epi32(x2, 16), _mm256_slli_epi32(n2, 15));
    return emit;
}

inline __m256i v_word32(__m256i& x1, __m256i& x2, __m256i mask) {
    const __m256i lo = v_step16(x1, x2, mask);
    const __m256i hi = v_step16(x1, x2, mask);
    return _mm256_or_si256(lo, _mm256_slli_epi32(hi, 16));
}

// 8x8 transpose of 32-bit elements: afterwards v[L] holds lane L's eight words
// in emission order. Amortised over 2048 generated bits.
inline void transpose8x8_epi32(__m256i v[8]) {
    const __m256i a0 = _mm256_unpacklo_epi32(v[0], v[1]);
    const __m256i a1 = _mm256_unpackhi_epi32(v[0], v[1]);
    const __m256i a2 = _mm256_unpacklo_epi32(v[2], v[3]);
    const __m256i a3 = _mm256_unpackhi_epi32(v[2], v[3]);
    const __m256i a4 = _mm256_unpacklo_epi32(v[4], v[5]);
    const __m256i a5 = _mm256_unpackhi_epi32(v[4], v[5]);
    const __m256i a6 = _mm256_unpacklo_epi32(v[6], v[7]);
    const __m256i a7 = _mm256_unpackhi_epi32(v[6], v[7]);

    const __m256i b0 = _mm256_unpacklo_epi64(a0, a2);
    const __m256i b1 = _mm256_unpackhi_epi64(a0, a2);
    const __m256i b2 = _mm256_unpacklo_epi64(a1, a3);
    const __m256i b3 = _mm256_unpackhi_epi64(a1, a3);
    const __m256i b4 = _mm256_unpacklo_epi64(a4, a6);
    const __m256i b5 = _mm256_unpackhi_epi64(a4, a6);
    const __m256i b6 = _mm256_unpacklo_epi64(a5, a7);
    const __m256i b7 = _mm256_unpackhi_epi64(a5, a7);

    v[0] = _mm256_permute2x128_si256(b0, b4, 0x20);
    v[1] = _mm256_permute2x128_si256(b1, b5, 0x20);
    v[2] = _mm256_permute2x128_si256(b2, b6, 0x20);
    v[3] = _mm256_permute2x128_si256(b3, b7, 0x20);
    v[4] = _mm256_permute2x128_si256(b0, b4, 0x31);
    v[5] = _mm256_permute2x128_si256(b1, b5, 0x31);
    v[6] = _mm256_permute2x128_si256(b2, b6, 0x31);
    v[7] = _mm256_permute2x128_si256(b3, b7, 0x31);
}

// One complete 2048-bit group: 8 lanes x 256 bits, lane L starting L*256 steps
// after (s1, s2), written as 8 x 32 bytes at dst + L*32. Used for the trailing
// partial group so that no part of the output falls back to scalar code.
void avx2_one_group(uint8_t* dst, uint32_t s1, uint32_t s2) {
    static const Mat j1 = mat_pow(step_matrix(kTapX1), kLaneGroupBits);
    static const Mat j2 = mat_pow(step_matrix(kTapX2), kLaneGroupBits);

    uint32_t init1[8], init2[8];
    init1[0] = s1;
    init2[0] = s2;
    for (unsigned L = 1; L < 8; ++L) {
        init1[L] = mat_apply(j1, init1[L - 1]);
        init2[L] = mat_apply(j2, init2[L - 1]);
    }

    __m256i x1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(init1));
    __m256i x2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(init2));
    const __m256i mask = _mm256_set1_epi32(0xFFFF);

    __m256i v[8];
    for (unsigned k = 0; k < 8; ++k) v[k] = v_word32(x1, x2, mask);
    transpose8x8_epi32(v);
    for (unsigned L = 0; L < 8; ++L)
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + L * 32), v[L]);
}

void prbs_avx2(uint8_t* out, size_t num_bits, const Seed& sd) {
    const size_t groups = num_bits / kGroupBits;
    const size_t vec_bits = groups * kGroupBits;

    // State at bit offset vec_bits, needed by the scalar tail. Lane 7 walks
    // exactly up to that point, so it is read back from the vector rather than
    // recomputed with another matrix power.
    uint32_t tail1 = sd.x1, tail2 = sd.x2;

    if (groups != 0) {
        const size_t lane_bits = groups * kLaneGroupBits;
        const size_t lane_bytes = lane_bits / 8;

        // Lane L starts L * lane_bits steps into the sequence.
        const Mat j1 = mat_pow(step_matrix(kTapX1), lane_bits);
        const Mat j2 = mat_pow(step_matrix(kTapX2), lane_bits);
        uint32_t init1[8], init2[8];
        init1[0] = sd.x1;
        init2[0] = sd.x2;
        for (unsigned L = 1; L < 8; ++L) {
            init1[L] = mat_apply(j1, init1[L - 1]);
            init2[L] = mat_apply(j2, init2[L - 1]);
        }

        // Unaligned loads on purpose: this is setup code that runs once per
        // call, so there is nothing to gain from demanding 32-byte alignment
        // on a stack buffer.
        __m256i x1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(init1));
        __m256i x2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(init2));
        const __m256i mask = _mm256_set1_epi32(0xFFFF);

        for (size_t g = 0; g < groups; ++g) {
            __m256i v[8];
            for (unsigned k = 0; k < 8; ++k) v[k] = v_word32(x1, x2, mask);
            transpose8x8_epi32(v);
            for (unsigned L = 0; L < 8; ++L)
                _mm256_storeu_si256(
                    reinterpret_cast<__m256i*>(out + L * lane_bytes + g * 32), v[L]);
        }

        uint32_t last1[8], last2[8];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(last1), x1);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(last2), x2);
        tail1 = last1[7];
        tail2 = last2[7];
    }

    if (num_bits > vec_bits) {
        // Generate one more whole group and keep the bits that are wanted. At
        // most 2047 bits are discarded, which is cheaper than dropping to a
        // scalar tail -- and keeps every generated bit on the AVX2 path.
        uint8_t scratch[kGroupBits / 8];
        avx2_one_group(scratch, tail1, tail2);
        std::memcpy(out + vec_bits / 8, scratch, (num_bits - vec_bits + 7) / 8);
    }
}

}  // namespace

uint32_t pusch_cinit(uint16_t rnti, uint16_t nid) {
    return ((static_cast<uint32_t>(rnti) << 15) + nid) & kMask31;
}

void generate_prbs(uint8_t* out, size_t num_bits, uint32_t cinit, Impl impl) {
    if (num_bits == 0) return;
    std::memset(out, 0, (num_bits + 7) / 8);

    const Seed sd = seed_after_nc(cinit);
    switch (impl) {
        case Impl::Scalar:
            prbs_scalar(out, num_bits, sd);
            break;
        case Impl::Wordwise:
            prbs_wordwise_from(out, num_bits, sd.x1, sd.x2);
            break;
        case Impl::Avx2:
            prbs_avx2(out, num_bits, sd);
            break;
    }

    // Contract: bits above num_bits in the final byte are always clear.
    const unsigned rem = num_bits & 7u;
    if (rem != 0) out[(num_bits + 7) / 8 - 1] &= static_cast<uint8_t>((1u << rem) - 1u);
}

}  // namespace pusch
