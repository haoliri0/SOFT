#ifndef STN_CUDA_WRITE_HPP
#define STN_CUDA_WRITE_HPP

#include <iostream>
#include <streambuf>
#include <utility>
#include "./simulator.hpp"
#include "./exceptions.hpp"

using namespace StnCuda;

// indent

const std::string default_indent = "  ";

class IndentedStreamBuf final : public std::streambuf {
public:
    std::streambuf *underlying;
    const std::string indent;
    bool active = true;
    IndentedStreamBuf(std::streambuf *const underlying, std::string indent)
        : underlying(underlying), indent(std::move(indent)) {}
protected:
    int overflow(const int c) override {
        constexpr auto eof = std::char_traits<char>::eof();
        if (c == eof)
            return c;

        // 如果在行首，先写入缩进
        if (active) {
            active = false;
            for (const char ch: indent)
                if (underlying->sputc(ch) == eof)
                    return eof;
        }

        // 如果是换行符，标记下一行需要缩进
        if (c == '\n')
            active = true;

        // 输出字符
        return underlying->sputc(static_cast<char>(c));
    }

    int sync() override {
        return underlying->pubsync();
    }
};

static
void with_indent(
    std::ostream &ostream,
    const std::string &indent,
    const std::function<void()> &block
) noexcept {
    IndentedStreamBuf indented_buf{ostream.rdbuf(), indent};
    ostream.rdbuf(&indented_buf);
    block();
    ostream.rdbuf(indented_buf.underlying);
}

static
void with_indent(
    std::ostream &ostream,
    const std::function<void()> &block
) noexcept {
    with_indent(ostream, default_indent, block);
}

// writing

template<typename T>
static
void write(std::ostream &ostream, T value) {
    ostream << value;
}

template<typename K, typename V>
static
void write_kv(std::ostream &ostream, K &key, V value) {
    write(ostream, key);
    write(ostream, ": ");
    write(ostream, value);
    write(ostream, "\n");
}

static
void write_bit(std::ostream &ostream, const Bit value) {
    write(ostream, !value ? "0" : "1");
}

static
void write_bits(std::ostream &ostream, const Bit *bits, const unsigned int n) {
    for (int i = 0; i < n; ++i) {
        write_bit(ostream, bits[i]);
        write(ostream, " ");
    }
}

static
void write_sign(std::ostream &ostream, const Bit sign) {
    write(ostream, !sign ? "+" : "-");
}

static
void write_phase(std::ostream &ostream, Phs phase) {
    phase %= 4;
    if (phase == 0) write(ostream, "+1");
    if (phase == 1) write(ostream, "+i");
    if (phase == 2) write(ostream, "-1");
    if (phase == 3) write(ostream, "-i");
}

static
void write_bst(std::ostream &ostream, Bst bst, const unsigned int n) {
    for (int i = 0; i < n; ++i) {
        write_bit(ostream, bst % 2);
        bst >>= 1;
    }
}

static
void write_amp(std::ostream &ostream, const Amp amp) {
    const auto real = amp.real();
    const auto imag = amp.imag();
    write_sign(ostream, real < 0);
    write(ostream, abs(real));
    write_sign(ostream, imag < 0);
    write(ostream, abs(imag));
    write(ostream, "i");
}

static
void write_cuda_error(std::ostream &ostream, const cudaError error) {
    write(ostream, cudaGetErrorName(error));
    write(ostream, "\n");
    write(ostream, cudaGetErrorString(error));
    write(ostream, "\n");
}


static
void write_pauli(std::ostream &ostream, const Bit x, const Bit z) {
    if (!x && !z) write(ostream, "I");
    if (x && !z) write(ostream, "X");
    if (!x && z) write(ostream, "Z");
    if (x && z) write(ostream, "Y");
}

static
void write_pauli_row(std::ostream &ostream, const PauliRowPtr &ptr) {
    for (Qid qubit_i = 0; qubit_i < ptr.qubits_n; ++qubit_i) {
        const Bit x = *ptr.get_x_ptr(qubit_i);
        const Bit z = *ptr.get_z_ptr(qubit_i);
        write_pauli(ostream, x, z);
    }
}

static
void write_table_row(std::ostream &ostream, const TableRowPtr &ptr) {
    write_sign(ostream, *ptr.get_sign_ptr());
    write_pauli_row(ostream, ptr.get_pauli_ptr());
    write(ostream, "\n");
}

static
void write_table(std::ostream &ostream, const TablePtr &ptr) {
    write(ostream, "table: |\n");
    with_indent(ostream, [&] {
        const Qid qubits_n = ptr.qubits_n;
        const Qid rows_n = 2 * qubits_n;
        for (Qid row_i = 0; row_i < rows_n; ++row_i)
            write_table_row(ostream, ptr.get_row_ptr(row_i));
    });
}

static
void write_pivot(std::ostream &ostream, const Qid pivot) {
    if (pivot == NullPivot) {
        write(ostream, "null");
    } else {
        write(ostream, pivot);
    }
}

static
void write_decomp(std::ostream &ostream, const DecompPtr ptr) {
    write(ostream, "decomposed:\n");
    with_indent(ostream, [&] {
        write(ostream, "bits: ");
        write_bits(ostream, ptr.get_bits_ptr(), 2 * ptr.qubits_n);
        printf("\n");

        write(ostream, "pivot: ");
        write_pivot(ostream, *ptr.get_pivot_ptr());
        printf("\n");

        write(ostream, "row: ");
        write_phase(ostream, *ptr.get_phase_ptr());
        write(ostream, " * ");
        write_pauli_row(ostream, ptr.get_pauli_ptr());
        printf("\n");
    });
}

static
void write_entry(std::ostream &ostream, const EntriesPtr ptr, const Eid entry_i) {
    const Bst bst = *ptr.get_bst_ptr(entry_i);
    const Amp amp = *ptr.get_amp_ptr(entry_i);
    write_bst(ostream, bst, ptr.qubits_n);
    write(ostream, ": ");
    write_amp(ostream, amp);
    write(ostream, "\n");
}

static
void write_entries(std::ostream &ostream, const EntriesPtr ptr, const bool full) {
    write(ostream, "entries:\n");
    with_indent(ostream, [&] {
        if (full) {
            const Eid entries_m = ptr.entries_m;
            write_kv(ostream, "entries_m", entries_m);
            for (Eid entry_i = 0; entry_i < entries_m; ++entry_i)
                write_entry(ostream, ptr, entry_i);
        } else {
            const Eid entries_n = *ptr.get_entries_n_ptr();
            write_kv(ostream, "entries_n", entries_n);
            for (Eid entry_i = 0; entry_i < entries_n; ++entry_i)
                write_entry(ostream, ptr, entry_i);
        }
    });
}

static
void write_work(std::ostream &ostream, const WorkPtr ptr) {
    write(ostream, "work:\n");
    with_indent(ostream, [&] {
        write_kv(ostream, "int", *ptr.get_int_ptr());
        write_kv(ostream, "flt: ", *ptr.get_flt_ptr());
        write_kv(ostream, "err: ", *ptr.get_err_ptr());
    });
}

static
void write_memory(std::ostream &ostream, const MemoryPtr ptr) {
    write(ostream, "memory:\n");
    with_indent(ostream, [&] {
        write(ostream, "ints:\n");
        with_indent(ostream, [&] {
            for (Mid mem_i = 0; mem_i < ptr.mem_ints_m; ++mem_i) {
                write(ostream, "- ");
                write(ostream, *ptr.get_int_ptr(mem_i));
                write(ostream, "\n");
            }
        });
        write(ostream, "flts:\n");
        with_indent(ostream, [&] {
            for (Mid mem_i = 0; mem_i < ptr.mem_flts_m; ++mem_i) {
                write(ostream, "- ");
                write(ostream, *ptr.get_flt_ptr(mem_i));
                write(ostream, "\n");
            }
        });
    });
}


static
void write_shot_state(std::ostream &ostream, const ShotStatePtr &ptr) {
    write_table(ostream, ptr.get_table_ptr());
    write_decomp(ostream, ptr.get_decomp_ptr());
    write_entries(ostream, ptr.get_entries_ptr(), false);
    write_work(ostream, ptr.get_work_ptr());
    write_memory(ostream, ptr.get_memory_ptr());
}

static
void write_shots_state(std::ostream &ostream, const ShotsStatePtr &ptr) {
    for (Sid shot_i = 0; shot_i < ptr.shots_n; ++shot_i) {
        write(ostream, "shot_");
        write(ostream, shot_i);
        write(ostream, ": ");
        write(ostream, "\n");
        with_indent(ostream, [&] {
            write_shot_state(ostream, ptr.get_shot_ptr(shot_i));
        });
    }
}

static
void write_simulator_args(std::ostream &ostream, const SimulatorArgs &args) {
    write(ostream, "args:\n");
    with_indent(ostream, [&] {
        write_kv(ostream, "shot_i", args.shot_i);
        write_kv(ostream, "shots_n", args.shots_n);
        write_kv(ostream, "qubits_n", args.qubits_n);
        write_kv(ostream, "entries_m", args.entries_m);
        write_kv(ostream, "mem_ints_m", args.mem_ints_m);
        write_kv(ostream, "mem_flts_m", args.mem_flts_m);
        write_kv(ostream, "epsilon", args.epsilon);
        write_kv(ostream, "seed", args.seed);
    });
}

#endif
