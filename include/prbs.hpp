// Length-31 Gold sequence generator - ETSI TS 38.211 v15.3.0, Section 5.2.1.
#ifndef PUSCH_PRBS_HPP
#define PUSCH_PRBS_HPP

#include <cstddef>
#include <cstdint>

namespace pusch {

// Avx2 is the implementation this design ships: every bit-wise operation of the
// sequence generator runs on AVX2 instructions. Scalar and Wordwise exist only
// as reference points for the efficiency comparison and as correctness oracles
// for the self-test -- they are not a production path and are never selected
// automatically.
enum class Impl {
    Scalar,    // bit-serial, literal transcription of the spec recurrences
    Wordwise,  // 24 bits per iteration on plain 32-bit registers
    Avx2       // 8 lanes x 32 bits per iteration, VPXOR/VPSRLD/VPSLLD/VPAND/VPOR
};

// c_init for PUSCH per TS 38.211 Section 6.3.1.1:
//   c_init = n_RNTI * 2^15 + n_ID      (n_ID = dataScramblingIdentityPUSCH,
//                                       or the cell id when not configured)
uint32_t pusch_cinit(uint16_t rnti, uint16_t nid);

// Writes num_bits of the Gold sequence c(0..num_bits-1) into out.
//
// Bit packing is LSB-first: c(i) lands at (out[i / 8] >> (i % 8)) & 1.
// out must have room for (num_bits + 7) / 8 bytes; unused trailing bits of the
// final byte are cleared. All three Impl values produce identical output.
void generate_prbs(uint8_t* out, size_t num_bits, uint32_t cinit, Impl impl);

}  // namespace pusch

#endif
