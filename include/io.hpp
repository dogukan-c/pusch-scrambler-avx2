#ifndef PUSCH_IO_HPP
#define PUSCH_IO_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace pusch {

// A bit stream packed LSB-first, plus its exact length in bits.
// bytes.size() == (num_bits + 7) / 8.
struct BitBuffer {
    std::vector<uint8_t> bytes;
    size_t num_bits = 0;
};

// Text format: whitespace-separated two-digit hex bytes.
// '#' begins a comment that runs to end of line. A "# bits=N" comment sets the
// exact bit count, which is how lengths that are not a multiple of 8 survive a
// round trip; without it the length is bytes * 8.
bool read_bits(const std::string& path, BitBuffer& buf, std::string& err);

// header, if non-empty, is written as a leading comment line.
bool write_bits(const std::string& path, const BitBuffer& buf, const std::string& header,
                std::string& err);

}  // namespace pusch

#endif
