#include "scrambler.hpp"

#include <immintrin.h>

namespace pusch {
namespace {

// Clears the bits above num_bits in the final partial byte, so a buffer never
// carries meaning past its declared length.
inline void mask_tail(uint8_t* out, size_t num_bits) {
    const unsigned rem = num_bits & 7u;
    if (rem != 0) out[(num_bits + 7) / 8 - 1] &= static_cast<uint8_t>((1u << rem) - 1u);
}

}  // namespace

void scramble_avx2(uint8_t* out, const uint8_t* in, const uint8_t* seq, size_t num_bits) {
    const size_t n = (num_bits + 7) / 8;
    size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(in + i));
        const __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(seq + i));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i), _mm256_xor_si256(a, b));
    }
    for (; i < n; ++i) out[i] = static_cast<uint8_t>(in[i] ^ seq[i]);
    mask_tail(out, num_bits);
}

void scramble_scalar(uint8_t* out, const uint8_t* in, const uint8_t* seq, size_t num_bits) {
    const size_t n = (num_bits + 7) / 8;
    for (size_t i = 0; i < n; ++i) out[i] = static_cast<uint8_t>(in[i] ^ seq[i]);
    mask_tail(out, num_bits);
}

}  // namespace pusch
