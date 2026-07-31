#include "core/common.hpp"

#include "core/internal.hpp"

namespace symft {
using namespace detail;

Error::Error(const std::string& message) : std::runtime_error(message) {}

int checked_nqubits(int nqubits) {
    if (nqubits < 0) {
        fail("number of qubits must be nonnegative");
    }
    return nqubits;
}

int nwords_for(int nqubits) {
    return (checked_nqubits(nqubits) + kWordBits - 1) / kWordBits;
}

std::size_t bit_word_count(int nbits) {
    if (nbits < 0) {
        fail("number of bits must be nonnegative");
    }
    return static_cast<std::size_t>((nbits + kWordBits - 1) / kWordBits);
}

bool packed_bit(const std::vector<std::uint64_t>& words, int bit_index) {
    if (bit_index < 0) {
        fail("bit index must be nonnegative");
    }
    const std::size_t word = static_cast<std::size_t>(bit_index / kWordBits);
    return word < words.size() && (words[word] & bit_mask(bit_index)) != 0;
}

void set_packed_bit(std::vector<std::uint64_t>& words, int bit_index, bool value) {
    if (bit_index < 0) {
        fail("bit index must be nonnegative");
    }
    const std::size_t word = static_cast<std::size_t>(bit_index / kWordBits);
    if (word >= words.size()) {
        words.resize(word + 1, 0);
    }
    if (value) {
        words[word] |= bit_mask(bit_index);
    } else {
        words[word] &= ~bit_mask(bit_index);
    }
}

std::vector<std::uint64_t> packed_bits(std::initializer_list<bool> bits) {
    return packed_bits(std::vector<bool>(bits));
}

std::vector<std::uint64_t> packed_bits(const std::vector<bool>& bits) {
    std::vector<std::uint64_t> out(bit_word_count(static_cast<int>(bits.size())), 0);
    int bit_index = 0;
    for (bool bit : bits) {
        if (bit) {
            set_packed_bit(out, bit_index);
        }
        ++bit_index;
    }
    return out;
}

} // namespace symft
