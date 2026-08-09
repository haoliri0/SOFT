#include "factored/pullback.hpp"

#include "core/internal.hpp"
#include "simd/simd.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <limits>
#include <queue>
#include <tuple>
#include <utility>

namespace symft::detail {

namespace {

struct LocalCliffordDescriptor {
    int arity = 0;
    void (*apply)(CliffordFrame&) = nullptr;
};

const LocalCliffordDescriptor& local_clifford(LocalCliffordGate gate) {
#define SYMFT_SINGLE_GATE(function_name) \
    LocalCliffordDescriptor { 1, [](CliffordFrame& frame) { function_name(frame, 0); } }
#define SYMFT_TWO_QUBIT_GATE(function_name) \
    LocalCliffordDescriptor { 2, [](CliffordFrame& frame) { function_name(frame, 0, 1); } }
    static const std::array<
        LocalCliffordDescriptor,
        static_cast<std::size_t>(LocalCliffordGate::Count)> descriptors{
        SYMFT_SINGLE_GATE(left_H),
        SYMFT_SINGLE_GATE(left_H_NXY),
        SYMFT_SINGLE_GATE(left_H_NXZ),
        SYMFT_SINGLE_GATE(left_H_NYZ),
        SYMFT_SINGLE_GATE(left_H_XY),
        SYMFT_SINGLE_GATE(left_H_YZ),
        SYMFT_SINGLE_GATE(left_C_NXYZ),
        SYMFT_SINGLE_GATE(left_C_NZYX),
        SYMFT_SINGLE_GATE(left_C_XNYZ),
        SYMFT_SINGLE_GATE(left_C_XYNZ),
        SYMFT_SINGLE_GATE(left_C_XYZ),
        SYMFT_SINGLE_GATE(left_C_ZNYX),
        SYMFT_SINGLE_GATE(left_C_ZYNX),
        SYMFT_SINGLE_GATE(left_C_ZYX),
        SYMFT_SINGLE_GATE(left_S),
        SYMFT_SINGLE_GATE(left_SDG),
        SYMFT_SINGLE_GATE(left_SQRT_X),
        SYMFT_SINGLE_GATE(left_SQRT_X_DAG),
        SYMFT_SINGLE_GATE(left_SQRT_Y),
        SYMFT_SINGLE_GATE(left_SQRT_Y_DAG),
        SYMFT_SINGLE_GATE(left_X),
        SYMFT_SINGLE_GATE(left_Y),
        SYMFT_SINGLE_GATE(left_Z),
        SYMFT_TWO_QUBIT_GATE(left_CX),
        SYMFT_TWO_QUBIT_GATE(left_CY),
        SYMFT_TWO_QUBIT_GATE(left_CZ),
        SYMFT_TWO_QUBIT_GATE(left_SWAP),
        SYMFT_TWO_QUBIT_GATE(left_CXSWAP),
        SYMFT_TWO_QUBIT_GATE(left_CZSWAP),
        SYMFT_TWO_QUBIT_GATE(left_ISWAP),
        SYMFT_TWO_QUBIT_GATE(left_ISWAP_DAG),
        SYMFT_TWO_QUBIT_GATE(left_SQRT_XX),
        SYMFT_TWO_QUBIT_GATE(left_SQRT_XX_DAG),
        SYMFT_TWO_QUBIT_GATE(left_SQRT_YY),
        SYMFT_TWO_QUBIT_GATE(left_SQRT_YY_DAG),
        SYMFT_TWO_QUBIT_GATE(left_SQRT_ZZ),
        SYMFT_TWO_QUBIT_GATE(left_SQRT_ZZ_DAG),
        SYMFT_TWO_QUBIT_GATE(left_SWAPCX),
        SYMFT_TWO_QUBIT_GATE(left_XCX),
        SYMFT_TWO_QUBIT_GATE(left_XCY),
        SYMFT_TWO_QUBIT_GATE(left_XCZ),
        SYMFT_TWO_QUBIT_GATE(left_YCX),
        SYMFT_TWO_QUBIT_GATE(left_YCY),
        SYMFT_TWO_QUBIT_GATE(left_YCZ),
    };
#undef SYMFT_SINGLE_GATE
#undef SYMFT_TWO_QUBIT_GATE

    const std::size_t index = static_cast<std::size_t>(gate);
    if (index >= descriptors.size()) {
        fail("invalid direct-pullback Clifford gate");
    }
    return descriptors[index];
}

struct LocalPauliMapEntry {
    std::uint8_t x = 0;
    std::uint8_t z = 0;
    std::uint8_t phase_delta = 0;
};

struct LocalCliffordTable {
    int arity = 0;
    std::array<LocalPauliMapEntry, 16> entries{};
    simd::PackedCliffordTransform packed;
};

unsigned table_body_from_packed_assignment(unsigned assignment, int arity) {
    unsigned x = assignment & 1u;
    unsigned z = (assignment >> 1) & 1u;
    if (arity == 2) {
        x |= ((assignment >> 2) & 1u) << 1;
        z |= ((assignment >> 3) & 1u) << 1;
    }
    return x | (z << arity);
}

unsigned packed_body_from_entry(const LocalPauliMapEntry& entry, int arity) {
    unsigned body = (entry.x & 1u) | ((entry.z & 1u) << 1);
    if (arity == 2) {
        body |= ((entry.x >> 1) & 1u) << 2;
        body |= ((entry.z >> 1) & 1u) << 3;
    }
    return body;
}

std::uint16_t algebraic_normal_form(std::uint16_t truth, unsigned variables) {
    const unsigned assignments = 1u << variables;
    for (unsigned variable = 0; variable < variables; ++variable) {
        for (unsigned assignment = 0; assignment < assignments; ++assignment) {
            if ((assignment & (1u << variable)) != 0 &&
                (truth & (std::uint16_t{1} << (assignment ^ (1u << variable)))) != 0) {
                truth ^= std::uint16_t{1} << assignment;
            }
        }
    }
    return truth;
}

simd::PackedCliffordTransform packed_transform(const LocalCliffordTable& table) {
    simd::PackedCliffordTransform transform;
    transform.arity = static_cast<std::uint8_t>(table.arity);
    const unsigned variables = static_cast<unsigned>(2 * table.arity);
    const unsigned assignments = 1u << variables;
    std::uint16_t phase_low_truth = 0;
    std::uint16_t phase_high_truth = 0;

    for (unsigned input = 0; input < variables; ++input) {
        const auto& entry = table.entries[
            table_body_from_packed_assignment(1u << input, table.arity)];
        const unsigned output = packed_body_from_entry(entry, table.arity);
        for (unsigned output_bit = 0; output_bit < variables; ++output_bit) {
            if ((output & (1u << output_bit)) != 0) {
                transform.output_masks[output_bit] |=
                    static_cast<std::uint8_t>(1u << input);
            }
        }
    }

    for (unsigned assignment = 0; assignment < assignments; ++assignment) {
        const auto& entry = table.entries[
            table_body_from_packed_assignment(assignment, table.arity)];
        const unsigned output = packed_body_from_entry(entry, table.arity);
        unsigned expected = 0;
        for (unsigned output_bit = 0; output_bit < variables; ++output_bit) {
            if ((std::popcount(
                    static_cast<unsigned>(transform.output_masks[output_bit]) & assignment) &
                 1) != 0) {
                expected |= 1u << output_bit;
            }
        }
        if (output != expected) {
            fail("Clifford Pauli-body map is not linear");
        }
        if ((entry.phase_delta & 1u) != 0) {
            phase_low_truth |= std::uint16_t{1} << assignment;
        }
        if ((entry.phase_delta & 2u) != 0) {
            phase_high_truth |= std::uint16_t{1} << assignment;
        }
    }
    transform.phase_low_anf = algebraic_normal_form(phase_low_truth, variables);
    transform.phase_high_anf = algebraic_normal_form(phase_high_truth, variables);
    return transform;
}

const LocalCliffordTable& pullback_table(LocalCliffordGate gate) {
    static const auto tables = [] {
        std::array<LocalCliffordTable, static_cast<std::size_t>(LocalCliffordGate::Count)> out{};
        for (std::size_t gate_index = 0; gate_index < out.size(); ++gate_index) {
            const auto gate = static_cast<LocalCliffordGate>(gate_index);
            const auto& descriptor = local_clifford(gate);
            auto& table = out[gate_index];
            table.arity = descriptor.arity;
            CliffordFrame frame(table.arity);
            descriptor.apply(frame);
            const unsigned body_count = 1u << (2 * table.arity);
            for (unsigned body = 0; body < body_count; ++body) {
                const unsigned x = body & ((1u << table.arity) - 1u);
                const unsigned z = body >> table.arity;
                PauliString input(table.arity);
                for (int q = 0; q < table.arity; ++q) {
                    input.set_xbit(q, ((x >> q) & 1u) != 0);
                    input.set_zbit(q, ((z >> q) & 1u) != 0);
                }
                input.set_phase(std::popcount(x & z));
                const PauliString output = preimage(frame, input);
                auto& entry = table.entries[body];
                for (int q = 0; q < table.arity; ++q) {
                    entry.x |= static_cast<std::uint8_t>(output.xbit(q)) << q;
                    entry.z |= static_cast<std::uint8_t>(output.zbit(q)) << q;
                }
                entry.phase_delta = static_cast<std::uint8_t>(
                    (output.phase_exponent() - input.phase_exponent()) & 3);
            }
            table.packed = packed_transform(table);
        }
        return out;
    }();
    const auto index = static_cast<std::size_t>(gate);
    if (index >= tables.size()) {
        fail("invalid direct-pullback Clifford gate");
    }
    return tables[index];
}

void apply_local_pullback(
    PauliString& pauli,
    LocalCliffordGate gate,
    int q0,
    int q1) {
    const auto& table = pullback_table(gate);
    const std::array<int, 2> qubits{q0, q1};
    unsigned x = 0;
    unsigned z = 0;
    for (int local = 0; local < table.arity; ++local) {
        x |= static_cast<unsigned>(pauli.xbit(qubits[local])) << local;
        z |= static_cast<unsigned>(pauli.zbit(qubits[local])) << local;
    }
    const auto& mapped = table.entries[x | (z << table.arity)];
    for (int local = 0; local < table.arity; ++local) {
        pauli.set_xbit(qubits[local], ((mapped.x >> local) & 1u) != 0);
        pauli.set_zbit(qubits[local], ((mapped.z >> local) & 1u) != 0);
    }
    pauli.phase_shift(mapped.phase_delta);
}

bool pauli_has_support(const PauliString& pauli, int q) {
    return pauli.xbit(q) || pauli.zbit(q);
}

class TransposedPauliBatch {
  public:
    TransposedPauliBatch(int nqubits, std::span<const PauliString> paulis)
        : nqubits_(nqubits),
          size_(paulis.size()),
          words_((size_ + 63) >> 6),
          body_columns_(2 * static_cast<std::size_t>(nqubits_) * words_, 0),
          phase_low_(words_, 0),
          phase_high_(words_, 0),
          symbolic_constant_(words_, 0),
          symbolic_conditions_(size_),
          anticommuting_scratch_(words_, 0) {
        for (std::size_t index = 0; index < size_; ++index) {
            const auto& pauli = paulis[index];
            if (pauli.nqubits != nqubits_) {
                fail("Pauli string dimension does not match direct-pullback batch");
            }
            const std::size_t batch_word = index >> 6;
            const std::uint64_t batch_mask = std::uint64_t{1} << (index & 63);
            for (std::size_t word = 0; word < pauli.x.size(); ++word) {
                std::uint64_t x_bits = pauli.x[word];
                std::uint64_t z_bits = pauli.z[word];
                while (x_bits) {
                    const std::size_t q =
                        word * 64 + static_cast<std::size_t>(trailing_zeros64(x_bits));
                    body_columns_[(2 * q) * words_ + batch_word] |= batch_mask;
                    x_bits &= x_bits - 1;
                }
                while (z_bits) {
                    const std::size_t q =
                        word * 64 + static_cast<std::size_t>(trailing_zeros64(z_bits));
                    body_columns_[(2 * q + 1) * words_ + batch_word] |= batch_mask;
                    z_bits &= z_bits - 1;
                }
            }
            if (pauli.phase_exponent() & 1) {
                phase_low_[batch_word] |= batch_mask;
            }
            if (pauli.phase_exponent() & 2) {
                phase_high_[batch_word] |= batch_mask;
            }
        }
    }

    std::uint64_t active_mask(std::size_t batch_word, std::size_t first_active) const {
        if (batch_word < (first_active >> 6)) {
            return 0;
        }
        std::uint64_t mask = ~std::uint64_t{0};
        if (batch_word == (first_active >> 6)) {
            mask &= ~std::uint64_t{0} << (first_active & 63);
        }
        if (batch_word + 1 == words_ && (size_ & 63) != 0) {
            mask &= (std::uint64_t{1} << (size_ & 63)) - 1;
        }
        return mask;
    }

    std::uint64_t x_word(int q, std::size_t batch_word) const {
        return body_columns_[2 * static_cast<std::size_t>(q) * words_ + batch_word];
    }

    std::uint64_t z_word(int q, std::size_t batch_word) const {
        return body_columns_[(2 * static_cast<std::size_t>(q) + 1) * words_ + batch_word];
    }

    void xor_symbolic_sign(
        std::size_t batch_word,
        std::uint64_t pending_mask,
        int condition) {
        if (pending_mask == 0) {
            return;
        }
        while (pending_mask) {
            const std::size_t pending_index =
                batch_word * 64 +
                static_cast<std::size_t>(trailing_zeros64(pending_mask));
            symbolic_conditions_[pending_index].push_back(condition);
            pending_mask &= pending_mask - 1;
        }
    }

    void seed_symbolic_sign(std::size_t index, const SymbolicBool& sign) {
        const std::size_t batch_word = index >> 6;
        const std::uint64_t pending_mask = std::uint64_t{1} << (index & 63);
        if (sign.constant) {
            symbolic_constant_[batch_word] |= pending_mask;
        }
        symbolic_conditions_[index] = sign.conditions;
    }

    void apply_gate(
        LocalCliffordGate gate,
        int q0,
        int q1,
        std::size_t first_active) {
        if (first_active >= size_) {
            return;
        }
        const auto& table = pullback_table(gate);
        auto* x0_column =
            body_columns_.data() + 2 * static_cast<std::size_t>(q0) * words_;
        auto* z0_column = x0_column + words_;
        auto* x1_column = table.arity == 2
                              ? body_columns_.data() +
                                    2 * static_cast<std::size_t>(q1) * words_
                              : nullptr;
        auto* z1_column = table.arity == 2 ? x1_column + words_ : nullptr;

        std::size_t full_word_begin = first_active >> 6;
        if ((first_active & 63) != 0) {
            const std::size_t word = full_word_begin++;
            const std::uint64_t active = active_mask(word, first_active);
            const std::uint64_t old_x0 = x0_column[word];
            const std::uint64_t old_z0 = z0_column[word];
            const std::uint64_t old_x1 = table.arity == 2 ? x1_column[word] : 0;
            const std::uint64_t old_z1 = table.arity == 2 ? z1_column[word] : 0;
            const std::uint64_t old_phase_low = phase_low_[word];
            const std::uint64_t old_phase_high = phase_high_[word];
            std::uint64_t new_x0 = old_x0;
            std::uint64_t new_z0 = old_z0;
            std::uint64_t new_x1 = old_x1;
            std::uint64_t new_z1 = old_z1;
            std::uint64_t new_phase_low = old_phase_low;
            std::uint64_t new_phase_high = old_phase_high;
            simd::scalar_table().apply_packed_clifford(
                &new_x0,
                &new_z0,
                table.arity == 2 ? &new_x1 : nullptr,
                table.arity == 2 ? &new_z1 : nullptr,
                &new_phase_low,
                &new_phase_high,
                1,
                table.packed);
            const auto merge_active = [active](std::uint64_t old_value, std::uint64_t new_value) {
                return (old_value & ~active) | (new_value & active);
            };
            x0_column[word] = merge_active(old_x0, new_x0);
            z0_column[word] = merge_active(old_z0, new_z0);
            if (table.arity == 2) {
                x1_column[word] = merge_active(old_x1, new_x1);
                z1_column[word] = merge_active(old_z1, new_z1);
            }
            phase_low_[word] = merge_active(old_phase_low, new_phase_low);
            phase_high_[word] = merge_active(old_phase_high, new_phase_high);
        }
        if (full_word_begin < words_) {
            simd::dispatch_table().apply_packed_clifford(
                x0_column + full_word_begin,
                z0_column + full_word_begin,
                table.arity == 2 ? x1_column + full_word_begin : nullptr,
                table.arity == 2 ? z1_column + full_word_begin : nullptr,
                phase_low_.data() + full_word_begin,
                phase_high_.data() + full_word_begin,
                words_ - full_word_begin,
                table.packed);
        }
    }

    void apply_conditional_sign(
        std::span<const std::uint32_t> column_indices,
        std::size_t first_active,
        int condition) {
        if (first_active >= size_ || column_indices.empty()) {
            return;
        }
        const std::size_t first_word = first_active >> 6;
        simd::dispatch_table().xor_packed_columns(
            anticommuting_scratch_.data() + first_word,
            body_columns_.data(),
            words_,
            column_indices.data(),
            column_indices.size(),
            first_word,
            words_ - first_word);
        for (std::size_t word = first_word; word < words_; ++word) {
            const std::uint64_t anticommuting =
                anticommuting_scratch_[word] & active_mask(word, first_active);
            xor_symbolic_sign(word, anticommuting, condition);
        }
    }

    std::vector<SymbolicPauliString> finish() {
        std::vector<SymbolicPauliString> out;
        out.reserve(size_);
        for (std::size_t index = 0; index < size_; ++index) {
            const std::size_t batch_word = index >> 6;
            const std::uint64_t batch_mask = std::uint64_t{1} << (index & 63);
            PauliString pauli(nqubits_);
            pauli.set_phase(
                static_cast<int>((phase_low_[batch_word] & batch_mask) != 0) |
                (static_cast<int>((phase_high_[batch_word] & batch_mask) != 0) << 1));
            SymbolicBool sign;
            sign.constant = (symbolic_constant_[batch_word] & batch_mask) != 0;
            out.emplace_back(std::move(pauli), std::move(sign));
        }
        for (int q = 0; q < nqubits_; ++q) {
            const std::size_t pauli_word = static_cast<std::size_t>(q) >> 6;
            const std::uint64_t pauli_mask = std::uint64_t{1} << (q & 63);
            for (std::size_t batch_word = 0; batch_word < words_; ++batch_word) {
                std::uint64_t x_bits = x_word(q, batch_word);
                std::uint64_t z_bits = z_word(q, batch_word);
                while (x_bits) {
                    const std::size_t index = batch_word * 64 +
                                              static_cast<std::size_t>(trailing_zeros64(x_bits));
                    out[index].pauli.x[pauli_word] |= pauli_mask;
                    x_bits &= x_bits - 1;
                }
                while (z_bits) {
                    const std::size_t index = batch_word * 64 +
                                              static_cast<std::size_t>(trailing_zeros64(z_bits));
                    out[index].pauli.z[pauli_word] |= pauli_mask;
                    z_bits &= z_bits - 1;
                }
            }
        }
        for (std::size_t index = 0; index < size_; ++index) {
            out[index].sign = SymbolicBool(
                out[index].sign.constant,
                std::move(symbolic_conditions_[index]));
        }
        return out;
    }

  private:
    int nqubits_ = 0;
    std::size_t size_ = 0;
    std::size_t words_ = 0;
    std::vector<std::uint64_t> body_columns_;
    std::vector<std::uint64_t> phase_low_;
    std::vector<std::uint64_t> phase_high_;
    std::vector<std::uint64_t> symbolic_constant_;
    std::vector<std::vector<int>> symbolic_conditions_;
    std::vector<std::uint64_t> anticommuting_scratch_;
};

} // namespace

DirectPullbackFrame::DirectPullbackFrame(int nqubits)
    : nqubits_(checked_nqubits(nqubits)),
      event_indices_by_qubit_(static_cast<std::size_t>(nqubits_)) {}

std::size_t DirectPullbackFrame::event_count() const {
    return events_.size();
}

void DirectPullbackFrame::append_clifford(LocalCliffordGate gate, int q) {
    if (local_clifford(gate).arity != 1) {
        fail("two-qubit Clifford gate requires two targets");
    }
    const int qi = check_qubit(nqubits_, q);
    Event event;
    event.kind = Event::Kind::Clifford;
    event.gate = gate;
    event.q0 = qi;
    const std::size_t event_index = events_.size();
    events_.push_back(event);
    event_indices_by_qubit_[static_cast<std::size_t>(qi)].push_back(event_index);
}

void DirectPullbackFrame::append_clifford(LocalCliffordGate gate, int q0, int q1) {
    if (local_clifford(gate).arity != 2) {
        fail("single-qubit Clifford gate requires one target");
    }
    const int q0i = check_qubit(nqubits_, q0);
    const int q1i = check_qubit(nqubits_, q1);
    if (q0i == q1i) {
        fail("two-qubit Clifford gate requires distinct qubits");
    }
    Event event;
    event.kind = Event::Kind::Clifford;
    event.gate = gate;
    event.q0 = q0i;
    event.q1 = q1i;
    const std::size_t event_index = events_.size();
    events_.push_back(event);
    event_indices_by_qubit_[static_cast<std::size_t>(q0i)].push_back(event_index);
    event_indices_by_qubit_[static_cast<std::size_t>(q1i)].push_back(event_index);
}

void DirectPullbackFrame::append_pauli(const PauliString& pauli, int condition) {
    if (pauli.nqubits != nqubits_) {
        fail("Pauli string dimension does not match direct-pullback frame");
    }
    if (condition <= 0) {
        fail("condition id must be positive");
    }
    const std::size_t begin = pauli_words_.size();
    for (std::size_t word = 0; word < pauli.x.size(); ++word) {
        if (pauli.x[word] != 0 || pauli.z[word] != 0) {
            pauli_words_.push_back(PauliWord{word, pauli.x[word], pauli.z[word]});
        }
    }
    const std::size_t count = pauli_words_.size() - begin;
    if (count == 0) {
        return;
    }
    Event event;
    event.kind = Event::Kind::ConditionalPauli;
    event.condition = condition;
    event.pauli_word_begin = begin;
    event.pauli_word_count = count;
    const std::size_t event_index = events_.size();
    for (std::size_t index = begin; index < begin + count; ++index) {
        const auto& word = pauli_words_[index];
        std::uint64_t support = word.x | word.z;
        while (support) {
            const std::size_t q =
                word.word * 64 + static_cast<std::size_t>(trailing_zeros64(support));
            event_indices_by_qubit_[q].push_back(event_index);
            support &= support - 1;
        }
    }
    events_.push_back(event);
}

void DirectPullbackFrame::append_single_qubit_pauli(
    int q,
    bool x,
    bool z,
    int condition) {
    const int qi = check_qubit(nqubits_, q);
    if (condition <= 0) {
        fail("condition id must be positive");
    }
    if (!x && !z) {
        return;
    }
    const std::size_t begin = pauli_words_.size();
    const std::size_t word = static_cast<std::size_t>(qi) >> 6;
    const std::uint64_t mask = std::uint64_t{1} << (qi & 63);
    pauli_words_.push_back(PauliWord{word, x ? mask : 0, z ? mask : 0});
    Event event;
    event.kind = Event::Kind::ConditionalPauli;
    event.condition = condition;
    event.pauli_word_begin = begin;
    event.pauli_word_count = 1;
    const std::size_t event_index = events_.size();
    events_.push_back(event);
    event_indices_by_qubit_[static_cast<std::size_t>(qi)].push_back(event_index);
}

bool DirectPullbackFrame::anticommutes_with(
    const PauliString& pauli,
    const Event& event) const {
    std::uint64_t parity = 0;
    const std::size_t end = event.pauli_word_begin + event.pauli_word_count;
    for (std::size_t index = event.pauli_word_begin; index < end; ++index) {
        const auto& word = pauli_words_[index];
        parity ^= (pauli.x[word.word] & word.z) ^ (pauli.z[word.word] & word.x);
    }
    return is_odd_popcount(parity);
}

SymbolicPauliString DirectPullbackFrame::pull_back(const PauliString& pauli) const {
    return pull_back(pauli, events_.size());
}

SymbolicPauliString DirectPullbackFrame::pull_back(
    const PauliString& pauli,
    std::size_t event_bound) const {
    if (pauli.nqubits != nqubits_) {
        fail("Pauli string dimension does not match direct-pullback frame");
    }
    if (event_bound > events_.size()) {
        fail("direct-pullback event bound is out of range");
    }
    using QueueEntry = std::tuple<std::size_t, int, std::size_t>;
    std::priority_queue<QueueEntry> queue;
    PauliString pulled_back = pauli;
    std::vector<int> conditions;

    const auto push_latest_before = [&](int q, std::size_t event_bound) {
        const auto& indices = event_indices_by_qubit_[static_cast<std::size_t>(q)];
        auto it = std::lower_bound(indices.begin(), indices.end(), event_bound);
        if (it != indices.begin()) {
            const std::size_t position = static_cast<std::size_t>((it - indices.begin()) - 1);
            queue.emplace(indices[position], q, position);
        }
    };

    for (std::size_t word = 0; word < pulled_back.x.size(); ++word) {
        std::uint64_t support = pulled_back.x[word] | pulled_back.z[word];
        while (support) {
            const int q = static_cast<int>(
                word * 64 + static_cast<std::size_t>(trailing_zeros64(support)));
            push_latest_before(q, event_bound);
            support &= support - 1;
        }
    }

    std::size_t last_processed_event = std::numeric_limits<std::size_t>::max();
    while (!queue.empty()) {
        const auto [event_index, q, position] = queue.top();
        queue.pop();
        const auto& event = events_[event_index];
        const bool first_visit = event_index != last_processed_event;
        bool q0_was_active = false;
        bool q1_was_active = false;
        if (first_visit) {
            if (event.kind == Event::Kind::Clifford) {
                q0_was_active = pauli_has_support(pulled_back, event.q0);
                q1_was_active = local_clifford(event.gate).arity == 2 &&
                                pauli_has_support(pulled_back, event.q1);
                apply_local_pullback(pulled_back, event.gate, event.q0, event.q1);
            } else if (anticommutes_with(pulled_back, event)) {
                conditions.push_back(event.condition);
            }
            last_processed_event = event_index;
        }

        if (pauli_has_support(pulled_back, q) && position != 0) {
            const auto& indices = event_indices_by_qubit_[static_cast<std::size_t>(q)];
            queue.emplace(indices[position - 1], q, position - 1);
        }

        if (first_visit && event.kind == Event::Kind::Clifford) {
            if (!q0_was_active && pauli_has_support(pulled_back, event.q0)) {
                push_latest_before(event.q0, event_index);
            }
            if (local_clifford(event.gate).arity == 2 && !q1_was_active &&
                pauli_has_support(pulled_back, event.q1)) {
                push_latest_before(event.q1, event_index);
            }
        }
    }

    return SymbolicPauliString(
        std::move(pulled_back),
        SymbolicBool(false, std::move(conditions)));
}

std::vector<SymbolicPauliString> DirectPullbackFrame::pull_back_batch(
    std::span<const PauliString> paulis,
    std::span<const std::size_t> event_bounds,
    std::span<const SymbolicBool> initial_signs) const {
    if (paulis.size() != event_bounds.size()) {
        fail("direct-pullback batch Pauli and event-bound counts differ");
    }
    if (!std::is_sorted(event_bounds.begin(), event_bounds.end())) {
        fail("direct-pullback batch event bounds must be nondecreasing");
    }
    if (!event_bounds.empty() && event_bounds.back() > events_.size()) {
        fail("direct-pullback batch event bound is out of range");
    }
    if (!initial_signs.empty() && initial_signs.size() != paulis.size()) {
        fail("direct-pullback batch initial-sign count differs from Pauli count");
    }

    // The sparse path needs only PauliWord masks. Decode the transposed column
    // IDs lazily so small direct-pullback workloads do not pay for batch-only
    // metadata while events are being collected.
    std::vector<std::size_t> anticommutation_column_offsets(events_.size() + 1, 0);
    std::vector<std::uint32_t> anticommutation_columns;
    anticommutation_columns.reserve(pauli_words_.size());
    for (std::size_t event_index = 0; event_index < events_.size(); ++event_index) {
        anticommutation_column_offsets[event_index] = anticommutation_columns.size();
        const auto& event = events_[event_index];
        if (event.kind != Event::Kind::ConditionalPauli) {
            continue;
        }
        const std::size_t end = event.pauli_word_begin + event.pauli_word_count;
        for (std::size_t index = event.pauli_word_begin; index < end; ++index) {
            const auto& word = pauli_words_[index];
            std::uint64_t x_bits = word.x;
            while (x_bits) {
                const std::size_t q =
                    word.word * 64 + static_cast<std::size_t>(trailing_zeros64(x_bits));
                anticommutation_columns.push_back(static_cast<std::uint32_t>(2 * q + 1));
                x_bits &= x_bits - 1;
            }
            std::uint64_t z_bits = word.z;
            while (z_bits) {
                const std::size_t q =
                    word.word * 64 + static_cast<std::size_t>(trailing_zeros64(z_bits));
                anticommutation_columns.push_back(static_cast<std::uint32_t>(2 * q));
                z_bits &= z_bits - 1;
            }
        }
    }
    anticommutation_column_offsets.back() = anticommutation_columns.size();

    TransposedPauliBatch batch(nqubits_, paulis);
    for (std::size_t index = 0; index < initial_signs.size(); ++index) {
        batch.seed_symbolic_sign(index, initial_signs[index]);
    }
    std::size_t first_active = paulis.size();
    for (std::size_t event_index = events_.size(); event_index-- > 0;) {
        while (first_active != 0 && event_bounds[first_active - 1] > event_index) {
            --first_active;
        }
        if (first_active == paulis.size()) {
            continue;
        }
        const auto& event = events_[event_index];
        if (event.kind == Event::Kind::Clifford) {
            batch.apply_gate(event.gate, event.q0, event.q1, first_active);
            continue;
        }
        const std::size_t column_begin =
            anticommutation_column_offsets[event_index];
        const std::size_t column_end =
            anticommutation_column_offsets[event_index + 1];
        batch.apply_conditional_sign(
            std::span<const std::uint32_t>(anticommutation_columns)
                .subspan(column_begin, column_end - column_begin),
            first_active,
            event.condition);
    }
    return batch.finish();
}

} // namespace symft::detail
