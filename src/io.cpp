#include "io.hpp"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>

namespace pusch {
namespace {

bool parse_hex_byte(const std::string& tok, uint8_t& out) {
    if (tok.size() != 2) return false;
    unsigned v = 0;
    for (char c : tok) {
        const unsigned char u = static_cast<unsigned char>(c);
        unsigned d;
        if (std::isdigit(u))
            d = static_cast<unsigned>(c - '0');
        else if (u >= 'a' && u <= 'f')
            d = static_cast<unsigned>(c - 'a') + 10;
        else if (u >= 'A' && u <= 'F')
            d = static_cast<unsigned>(c - 'A') + 10;
        else
            return false;
        v = (v << 4) | d;
    }
    out = static_cast<uint8_t>(v);
    return true;
}

}  // namespace

bool read_bits(const std::string& path, BitBuffer& buf, std::string& err) {
    std::ifstream file(path);
    if (!file) {
        err = "cannot open '" + path + "'";
        return false;
    }

    buf.bytes.clear();
    buf.num_bits = 0;
    size_t declared_bits = 0;
    unsigned lineno = 0;
    std::string line;

    while (std::getline(file, line)) {
        ++lineno;
        const size_t hash = line.find('#');
        if (hash != std::string::npos) {
            const size_t key = line.find("bits=", hash);
            if (key != std::string::npos)
                declared_bits = std::strtoull(line.c_str() + key + 5, nullptr, 10);
            line.erase(hash);
        }
        std::istringstream ls(line);
        std::string tok;
        while (ls >> tok) {
            uint8_t b;
            if (!parse_hex_byte(tok, b)) {
                err = path + ":" + std::to_string(lineno) + ": expected a two-digit hex byte, got '" + tok + "'";
                return false;
            }
            buf.bytes.push_back(b);
        }
    }

    if (buf.bytes.empty()) {
        err = path + ": no data bytes found";
        return false;
    }

    const size_t avail = buf.bytes.size() * 8;
    if (declared_bits == 0) {
        buf.num_bits = avail;
    } else if (declared_bits > avail) {
        err = path + ": header declares bits=" + std::to_string(declared_bits) + " but only " +
              std::to_string(avail) + " bits of data are present";
        return false;
    } else {
        buf.num_bits = declared_bits;
        buf.bytes.resize((declared_bits + 7) / 8);
    }
    return true;
}

bool write_bits(const std::string& path, const BitBuffer& buf, const std::string& header,
                std::string& err) {
    std::ofstream file(path);
    if (!file) {
        err = "cannot write '" + path + "'";
        return false;
    }

    if (!header.empty()) file << "# " << header << "\n";
    file << "# bits=" << buf.num_bits << "\n";

    char hex[4];
    for (size_t i = 0; i < buf.bytes.size(); ++i) {
        std::snprintf(hex, sizeof hex, "%02X", buf.bytes[i]);
        if (i != 0) file << ((i % 16 == 0) ? '\n' : ' ');
        file << hex;
    }
    file << '\n';

    if (!file) {
        err = "write failed for '" + path + "'";
        return false;
    }
    return true;
}

}  // namespace pusch
