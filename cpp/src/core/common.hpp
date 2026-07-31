#pragma once

#include <complex>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <vector>

namespace symft {

using Complex = std::complex<double>;

class Error : public std::runtime_error {
public:
    explicit Error(const std::string& message);
};

int checked_nqubits(int nqubits);
int nwords_for(int nqubits);
std::size_t bit_word_count(int nbits);
bool packed_bit(const std::vector<std::uint64_t>& words, int bit_index);
void set_packed_bit(std::vector<std::uint64_t>& words, int bit_index, bool value = true);
std::vector<std::uint64_t> packed_bits(std::initializer_list<bool> bits);
std::vector<std::uint64_t> packed_bits(const std::vector<bool>& bits);

} // namespace symft
