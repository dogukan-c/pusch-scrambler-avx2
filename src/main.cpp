#include <x86intrin.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "io.hpp"
#include "prbs.hpp"
#include "scrambler.hpp"

namespace {

using pusch::BitBuffer;
using pusch::Impl;

constexpr uint16_t kMaxNid = 1023;  // dataScramblingIdentityPUSCH, TS 38.211 6.3.1.1

const char* kUsage =
    "5G NR PUSCH scrambler (ETSI TS 38.211 v15.3.0 Sec. 6.3.1.1), AVX2\n"
    "\n"
    "Usage:\n"
    "  pusch_scrambler scramble --in FILE --out FILE --rnti N --nid N [--impl I]\n"
    "  pusch_scrambler selftest\n"
    "  pusch_scrambler bench [--bits N] [--iters N]\n"
    "\n"
    "Options:\n"
    "  --in FILE     input bits, whitespace-separated hex bytes ('#' comments;\n"
    "                a '# bits=N' line gives an exact non-byte-aligned length)\n"
    "  --out FILE    scrambled output, same format\n"
    "  --rnti N      n_RNTI, 0..65535\n"
    "  --nid N       n_ID, 0..1023 (cell id, or dataScramblingIdentityPUSCH)\n"
    "  --impl I      avx2 (default, the shipped design) | wordwise | scalar\n"
    "                wordwise and scalar are reference baselines for the\n"
    "                efficiency comparison, not a production path\n"
    "  --bits N      benchmark payload size in bits (default: 1000000)\n"
    "  --iters N     benchmark repetitions, best-of (default: 200)\n"
    "\n"
    "Scrambling is an involution: running the output back through the same\n"
    "--rnti/--nid recovers the input.\n";

bool parse_impl(const std::string& s, Impl& out) {
    if (s == "scalar") out = Impl::Scalar;
    else if (s == "wordwise") out = Impl::Wordwise;
    else if (s == "avx2") out = Impl::Avx2;
    else return false;
    return true;
}

const char* impl_name(Impl i) {
    switch (i) {
        case Impl::Scalar: return "scalar";
        case Impl::Wordwise: return "wordwise";
        case Impl::Avx2: return "avx2";
    }
    return "?";
}

// Parses an unsigned decimal argument, rejecting junk and out-of-range values.
bool parse_uint(const char* s, unsigned long long lo, unsigned long long hi,
                unsigned long long& out) {
    if (s == nullptr || *s == '\0') return false;
    char* end = nullptr;
    errno = 0;
    const unsigned long long v = std::strtoull(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0' || v < lo || v > hi) return false;
    out = v;
    return true;
}

// ------------------------------------------------------------------ timing
struct Timing {
    double sec = 1e30;
    double tsc = 0;
};

template <class F>
Timing best_of(F&& f, unsigned iters) {
    Timing best;
    for (unsigned i = 0; i < iters; ++i) {
        _mm_lfence();
        const uint64_t c0 = __rdtsc();
        const auto t0 = std::chrono::steady_clock::now();
        f();
        const auto t1 = std::chrono::steady_clock::now();
        _mm_lfence();
        const uint64_t c1 = __rdtsc();
        const double s = std::chrono::duration<double>(t1 - t0).count();
        if (s < best.sec) {
            best.sec = s;
            best.tsc = static_cast<double>(c1 - c0);
        }
    }
    return best;
}

// ---------------------------------------------------------------- commands
int cmd_scramble(int argc, char** argv) {
    std::string in_path, out_path;
    unsigned long long rnti = 0, nid = 0;
    bool have_rnti = false, have_nid = false;
    Impl impl = Impl::Avx2;

    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        const bool has_val = (i + 1 < argc);
        if (a == "--in" && has_val) in_path = argv[++i];
        else if (a == "--out" && has_val) out_path = argv[++i];
        else if (a == "--rnti" && has_val) {
            if (!parse_uint(argv[++i], 0, 65535, rnti)) {
                std::fprintf(stderr, "error: --rnti must be 0..65535\n");
                return 2;
            }
            have_rnti = true;
        } else if (a == "--nid" && has_val) {
            if (!parse_uint(argv[++i], 0, kMaxNid, nid)) {
                std::fprintf(stderr, "error: --nid must be 0..%u\n", kMaxNid);
                return 2;
            }
            have_nid = true;
        } else if (a == "--impl" && has_val) {
            if (!parse_impl(argv[++i], impl)) {
                std::fprintf(stderr, "error: --impl must be scalar, wordwise or avx2\n");
                return 2;
            }
        } else {
            std::fprintf(stderr, "error: unexpected argument '%s'\n\n%s", a.c_str(), kUsage);
            return 2;
        }
    }

    if (in_path.empty() || out_path.empty() || !have_rnti || !have_nid) {
        std::fprintf(stderr, "error: --in, --out, --rnti and --nid are all required\n\n%s", kUsage);
        return 2;
    }

    BitBuffer buf;
    std::string err;
    if (!pusch::read_bits(in_path, buf, err)) {
        std::fprintf(stderr, "error: %s\n", err.c_str());
        return 1;
    }

    const uint32_t cinit = pusch::pusch_cinit(static_cast<uint16_t>(rnti), static_cast<uint16_t>(nid));
    std::vector<uint8_t> seq(buf.bytes.size());
    pusch::generate_prbs(seq.data(), buf.num_bits, cinit, impl);
    pusch::scramble_avx2(buf.bytes.data(), buf.bytes.data(), seq.data(), buf.num_bits);

    char header[160];
    std::snprintf(header, sizeof header, "PUSCH scrambled: rnti=%llu nid=%llu c_init=0x%08X impl=%s",
                  rnti, nid, cinit, impl_name(impl));
    if (!pusch::write_bits(out_path, buf, header, err)) {
        std::fprintf(stderr, "error: %s\n", err.c_str());
        return 1;
    }

    std::printf("%zu bits scrambled (rnti=%llu, nid=%llu, c_init=0x%08X, impl=%s) -> %s\n",
                buf.num_bits, rnti, nid, cinit, impl_name(impl), out_path.c_str());
    return 0;
}

int cmd_selftest() {
    // Cross-check every implementation over lengths that exercise the AVX2
    // group boundary (2048 bits), the 24-bit wordwise stride, and sub-byte tails.
    const size_t lengths[] = {1,   7,    8,    23,   24,   25,   31,    32,
                              255, 2047, 2048, 2049, 4096, 8460, 20000, 1000003};
    const uint16_t rntis[] = {0, 1, 466, 13, 65535};
    const uint16_t nids[] = {0, 270, 1023, 512, 1007};

    unsigned checks = 0, failures = 0;
    for (uint16_t rnti : rntis) {
        for (uint16_t nid : nids) {
            const uint32_t cinit = pusch::pusch_cinit(rnti, nid);
            for (size_t bits : lengths) {
                const size_t nb = (bits + 7) / 8;
                std::vector<uint8_t> a(nb), b(nb), c(nb);
                pusch::generate_prbs(a.data(), bits, cinit, Impl::Scalar);
                pusch::generate_prbs(b.data(), bits, cinit, Impl::Wordwise);
                pusch::generate_prbs(c.data(), bits, cinit, Impl::Avx2);
                ++checks;
                if (a != b || a != c) {
                    std::printf("FAIL prbs  rnti=%u nid=%u bits=%zu (%s)\n", rnti, nid, bits,
                                (a != b) ? "wordwise differs" : "avx2 differs");
                    ++failures;
                    continue;
                }

                // Scrambling must be an involution, and both kernels must agree.
                std::vector<uint8_t> data(nb), s1(nb), s2(nb), back(nb);
                for (size_t i = 0; i < nb; ++i)
                    data[i] = static_cast<uint8_t>((i * 131u + bits * 17u + rnti) & 0xFF);
                if (bits % 8) data[nb - 1] &= static_cast<uint8_t>((1u << (bits % 8)) - 1u);

                pusch::scramble_avx2(s1.data(), data.data(), a.data(), bits);
                pusch::scramble_scalar(s2.data(), data.data(), a.data(), bits);
                pusch::scramble_avx2(back.data(), s1.data(), a.data(), bits);
                ++checks;
                if (s1 != s2 || back != data) {
                    std::printf("FAIL xor   rnti=%u nid=%u bits=%zu (%s)\n", rnti, nid, bits,
                                (s1 != s2) ? "kernels differ" : "not an involution");
                    ++failures;
                }
            }
        }
    }

    // c_init formula, TS 38.211 6.3.1.1.
    ++checks;
    if (pusch::pusch_cinit(466, 270) != ((466u << 15) + 270u)) {
        std::printf("FAIL cinit\n");
        ++failures;
    }

    std::printf("%s: %u checks, %u failures\n", failures ? "FAILED" : "PASS", checks, failures);
    return failures ? 1 : 0;
}

int cmd_bench(int argc, char** argv) {
    unsigned long long bits = 1000000, iters = 200;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        const bool has_val = (i + 1 < argc);
        if (a == "--bits" && has_val) {
            if (!parse_uint(argv[++i], 1, 1ull << 32, bits)) {
                std::fprintf(stderr, "error: --bits must be 1..2^32\n");
                return 2;
            }
        } else if (a == "--iters" && has_val) {
            if (!parse_uint(argv[++i], 1, 1000000, iters)) {
                std::fprintf(stderr, "error: --iters must be 1..1000000\n");
                return 2;
            }
        } else {
            std::fprintf(stderr, "error: unexpected argument '%s'\n\n%s", a.c_str(), kUsage);
            return 2;
        }
    }

    const size_t nbits = static_cast<size_t>(bits);
    const size_t nbytes = (nbits + 7) / 8;
    const uint32_t cinit = pusch::pusch_cinit(466, 270);

    std::vector<uint8_t> seq(nbytes), data(nbytes), out(nbytes);
    for (size_t i = 0; i < nbytes; ++i) data[i] = static_cast<uint8_t>(i * 131u + 7u);

    // Warm-up: fault in the pages and settle the frequency before measuring.
    for (unsigned i = 0; i < 8; ++i) {
        pusch::generate_prbs(seq.data(), nbits, cinit, Impl::Avx2);
        pusch::scramble_avx2(out.data(), data.data(), seq.data(), nbits);
    }

    volatile uint8_t sink = 0;
    const double b = static_cast<double>(nbits);

    struct Row {
        const char* name;
        Timing t;
    };

    const unsigned it = static_cast<unsigned>(iters);
    // The bit-serial reference is ~100x slower; giving it the full iteration
    // count would dominate the run time without improving the estimate.
    const unsigned it_slow = std::max(3u, it / 20u);

    Row prbs[3] = {
        {"scalar", best_of([&] { pusch::generate_prbs(seq.data(), nbits, cinit, Impl::Scalar); }, it_slow)},
        {"wordwise", best_of([&] { pusch::generate_prbs(seq.data(), nbits, cinit, Impl::Wordwise); }, it)},
        {"avx2", best_of([&] { pusch::generate_prbs(seq.data(), nbits, cinit, Impl::Avx2); }, it)},
    };
    sink = static_cast<uint8_t>(sink ^ seq[nbytes - 1]);

    Row xorr[2] = {
        {"scalar", best_of([&] { pusch::scramble_scalar(out.data(), data.data(), seq.data(), nbits); }, it)},
        {"avx2", best_of([&] { pusch::scramble_avx2(out.data(), data.data(), seq.data(), nbits); }, it)},
    };
    sink = static_cast<uint8_t>(sink ^ out[nbytes - 1]);

    const Timing e2e = best_of(
        [&] {
            pusch::generate_prbs(seq.data(), nbits, cinit, Impl::Avx2);
            pusch::scramble_avx2(out.data(), data.data(), seq.data(), nbits);
        },
        it);
    sink = static_cast<uint8_t>(sink ^ out[0]);
    (void)sink;

    const double tsc_ghz = prbs[2].t.tsc / prbs[2].t.sec / 1e9;
    std::printf("Payload    : %zu bits (%zu bytes)\n", nbits, nbytes);
    std::printf("Statistic  : best of %u runs (%u for the bit-serial reference)\n", it, it_slow);
    std::printf("TSC rate   : %.3f GHz (invariant TSC, i.e. base clock - not the turbo core clock)\n\n", tsc_ghz);

    auto row = [&](const Row& r, const Timing& ref) {
        std::printf("  %-10s %10.3f us %9.3f %11.2f %10.1fx\n", r.name, r.t.sec * 1e6,
                    r.t.tsc / b, b / r.t.sec / 1e9, ref.sec / r.t.sec);
    };

    std::printf("PRBS generation (Gold sequence, TS 38.211 5.2.1)\n");
    std::printf("  %-10s %13s %9s %11s %11s\n", "impl", "time", "tsc/bit", "Gbit/s", "speedup");
    for (const Row& r : prbs) row(r, prbs[0].t);

    std::printf("\nScrambling (b XOR c)\n");
    std::printf("  %-10s %13s %9s %11s %11s\n", "impl", "time", "tsc/bit", "Gbit/s", "speedup");
    for (const Row& r : xorr) row(r, xorr[0].t);

    std::printf("\nEnd-to-end (generate + scramble, avx2)\n");
    std::printf("  %-10s %10.3f us %9.3f %11.2f\n", "total", e2e.sec * 1e6, e2e.tsc / b,
                b / e2e.sec / 1e9);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "%s", kUsage);
        return 2;
    }
    const std::string cmd = argv[1];
    if (cmd == "-h" || cmd == "--help" || cmd == "help") {
        std::printf("%s", kUsage);
        return 0;
    }
    if (cmd == "scramble") return cmd_scramble(argc, argv);
    if (cmd == "selftest") return cmd_selftest();
    if (cmd == "bench") return cmd_bench(argc, argv);

    std::fprintf(stderr, "error: unknown command '%s'\n\n%s", cmd.c_str(), kUsage);
    return 2;
}
