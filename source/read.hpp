#ifndef STN_CUDA_READ_HPP
#define STN_CUDA_READ_HPP

#include <istream>
#include <charconv>
#include <exception>
#include "./simulator.hpp"

static
bool match(const char *str, const char *seg) {
    while (true) {
        if (*str != *seg) return false;
        if (*seg == '\0') return true;
        str++;
        seg++;
    }
}

static
const char *match_head(const char *str, const char *seg) {
    while (true) {
        if (*seg == '\0') return str;
        if (*str != *seg) return nullptr;
        str++;
        seg++;
    }
}


class ParseException final : std::exception {
public:
    const std::errc ec;
    explicit ParseException(const std::errc ec) : ec(ec) {}
    [[nodiscard]] const char *what() const noexcept override {
        return std::make_error_code(ec).message().c_str();
    }
};

template<typename T>
static
void parse_value(const char *head, const char *tail, T &value) {
    const auto [ptr, ec] = std::from_chars(head, tail, value);
    if (ec != std::errc{}) throw ParseException(ec);
    if (ptr != tail) throw ParseException(std::errc::invalid_argument);
}

template<typename T>
static
void parse_value(const char *chars, T &value) {
    auto head = chars;
    auto tail = head + strlen(head);
    parse_value(head, tail, value);
}


static
bool is_linebreak(const int c) {
    return c == '\n' || c == '\r';
}

static
bool is_whitespace(const int c) {
    return c == ' ' || c == '\t';
}

static
void skip(std::istream &istream, const std::function<bool(char)> &cond) {
    while (istream.good()) {
        const int c = istream.peek();
        if (c == EOF) break;
        if (!cond(c)) break;
        istream.ignore();
    }
}

static
void ensure(std::istream &istream, const std::function<bool(char)> &cond, const bool consume) {
    const int c = consume ? istream.get() : istream.peek();
    if (istream.bad()) throw ParseException(std::errc::io_error);
    if (c == EOF || cond(c)) return;
    throw ParseException(std::errc::invalid_argument);
}


template<typename T>
static
void read_value(std::istream &istream, T &value) {
    skip(istream, is_whitespace);

    istream >> value;
    if (istream.bad()) throw ParseException(std::errc::io_error);
    if (istream.fail()) throw ParseException(std::errc::invalid_argument);
}

static
void read_value(std::istream &istream, Bit &arg) {
    unsigned int value;
    read_value(istream, value);
    if (value != 0 && value != 1) {
        fprintf(stderr, "Expected 0 or 1 for bits, got %u.\n", value);
        throw ParseException(std::errc::invalid_argument);
    }

    arg = value;
}

static
void read_value(std::istream &istream, std::string &value) {
    skip(istream, is_whitespace);

    istream >> value;
    if (istream.bad()) throw ParseException(std::errc::io_error);
    if (istream.fail()) {
        istream.clear(istream.rdstate() & ~std::ios::failbit);
        value.clear();
    }
}

template<typename Item>
static
void read_value(std::istream &, Array<Item, 0> &) {
    // nothing to do
}

template<typename Item, unsigned int n>
static
void read_value(std::istream &istream, Array<Item, n> &value) {
    read_value(istream, value.item);
    read_value(istream, value.tail);
}

template<typename Item>
static
void read_value(std::istream &istream, std::vector<Item> &value) {
    while (true) {
        Item item;
        try {
            read_value(istream, item);
        } catch (ParseException &exception) {
            if (exception.ec == std::errc::invalid_argument) {
                istream.clear(istream.rdstate() & ~std::ios::failbit);
                break;
            }
            throw;
        }
        value.push_back(item);
    }
    skip(istream, is_whitespace);
    ensure(istream, is_linebreak, false);
}

template<Mid m>
static
void read_value(std::istream &istream, ClassicalReduceArgs<m> &value) {
    std::vector<Mid> pointers;
    read_value(istream, pointers);

    const Mid n = pointers.size();
    if (n > m) {
        fprintf(stderr, "This op can hold at most %u arguments, got %u.\n", m, n);
        throw ParseException(std::errc::invalid_argument);
    }

    value.n = n;
    for (size_t i = 0; i < n; ++i)
        value.pointers.get(i) = pointers[i];
}

template<Mid m>
static
void read_value(std::istream &istream, ClassicalMatchArgs<m> &value) {
    std::vector<int> items;
    read_value(istream, items);

    const Mid n2 = items.size();
    if (n2 % 2 != 0) {
        fprintf(stderr, "This op expects 2*n arguments, got %u.\n", n2);
        throw ParseException(std::errc::invalid_argument);
    }
    const Mid n = n2 / 2;
    if (n > m) {
        fprintf(stderr, "This op can hold at most %u arguments, got %u.\n", 2 * m, 2 * n);
        throw ParseException(std::errc::invalid_argument);
    }
    for (size_t i = 0; i < n; ++i) {
        if (items[i] < 0) {
            fprintf(stderr, "This op Match expected positive integers for pointers, got %d.\n", items[i]);
            throw ParseException(std::errc::invalid_argument);
        }
    }

    value.n = n;
    for (size_t i = 0; i < n; ++i)
        value.pointers.get(i) = items[i];
    for (size_t i = 0; i < n; ++i)
        value.values.get(i) = items[n + i];
}


#endif
