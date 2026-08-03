// PUSCH scrambling - ETSI TS 38.211 v15.3.0, Section 6.3.1.1.
#ifndef PUSCH_SCRAMBLER_HPP
#define PUSCH_SCRAMBLER_HPP

#include <cstddef>
#include <cstdint>

namespace pusch {

// b~(i) = b(i) XOR c(i), over LSB-first packed bit buffers.
// Trailing bits of the final partial byte are cleared. in/out may alias.
void scramble_avx2(uint8_t* out, const uint8_t* in, const uint8_t* seq, size_t num_bits);

// Same result without intrinsics; kept as the benchmark baseline.
void scramble_scalar(uint8_t* out, const uint8_t* in, const uint8_t* seq, size_t num_bits);

}  // namespace pusch

#endif
