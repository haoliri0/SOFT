#ifndef STN_CUDA_READ_HPP
#define STN_CUDA_READ_HPP

#include <istream>

static
unsigned long parse_cli_ul(const char *s) {
    char *ptr;
    const unsigned long v = strtoul(s, &ptr, 10);

    if (errno != 0 || ptr == s || *ptr != '\0') {
        fprintf(stderr, "Failed to parse '%s' with errno=%d", s,errno);
        throw CliArgsException(CliArgsError::IllegalValue);
    }

    return v;
}

static
unsigned long long parse_cli_ull(const char *s) {
    char *ptr;
    const unsigned long long v = strtoull(s, &ptr, 10);

    if (errno != 0 || ptr == s || *ptr != '\0') {
        fprintf(stderr, "Failed to parse '%s' with errno=%d", s,errno);
        throw CliArgsException(CliArgsError::IllegalValue);
    }

    return v;
}

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

static
bool is_linebreak(const int c) {
    return c == '\n' || c == '\r';
}

static
bool is_whitespace(const int c) {
    return c == ' ' || c == '\t';
}

static
void skip_whitespace(std::istream &istream) {
    while (istream.good()) {
        const int c = istream.peek();
        if (c == EOF) break;
        if (!is_whitespace(c)) break;
        istream.ignore();
    }
}

static
void skip_whitespace_line(std::istream &istream) {
    skip_whitespace(istream);
    if (!istream.good()) return;

    const int c = istream.peek();
    if (c == EOF) return;
    if (!is_linebreak(c)) {
        istream.setstate(std::istream::failbit);
        return;
    }

    istream.ignore();
}

static
void read_word(std::istream &istream, size_t limit, char *buffer) {
    // 跳过空白字符（不包括换行符）
    skip_whitespace(istream);

    // 读取单词到缓冲区
    while (istream.good() && limit > 0) {
        const int c = istream.peek();
        if (c == EOF) break;
        if (is_whitespace(c) || is_linebreak(c)) break;
        *buffer = static_cast<char>(c);
        istream.ignore();
        buffer++;
        limit--;
    }

    // 检查是否达到缓冲区限制
    if (limit == 0) {
        istream.setstate(std::istream::failbit);
        return;
    }

    // 确保字符串以空字符结尾
    *buffer = '\0';
}

static
void read_arg(std::istream &istream, int &arg) {
    skip_whitespace(istream);
    if (istream.bad()) throw ExecException(ExecError::IOError);
    if (istream.fail()) throw ExecException(ExecError::IllegalArg);

    istream >> arg;
    if (istream.bad()) throw ExecException(ExecError::IOError);
    if (istream.fail()) throw ExecException(ExecError::IllegalArg);
}

static
void read_arg(std::istream &istream, Qid &arg) {
    skip_whitespace(istream);
    if (istream.bad()) throw ExecException(ExecError::IOError);
    if (istream.fail()) throw ExecException(ExecError::IllegalArg);

    istream >> arg;
    if (istream.bad()) throw ExecException(ExecError::IOError);
    if (istream.fail()) throw ExecException(ExecError::IllegalArg);
}

static
void read_arg(std::istream &istream, Flt &arg) {
    skip_whitespace(istream);
    if (istream.bad()) throw ExecException(ExecError::IOError);
    if (istream.fail()) throw ExecException(ExecError::IllegalArg);

    istream >> arg;
    if (istream.bad()) throw ExecException(ExecError::IOError);
    if (istream.fail()) throw ExecException(ExecError::IllegalArg);
}

static
void read_arg(std::istream &istream, Bit &arg) {
    skip_whitespace(istream);
    if (istream.bad()) throw ExecException(ExecError::IOError);
    if (istream.fail()) throw ExecException(ExecError::IllegalArg);

    unsigned int value;
    istream >> value;
    if (istream.bad()) throw ExecException(ExecError::IOError);
    if (istream.fail()) throw ExecException(ExecError::IllegalArg);

    if (value != 0 && value != 1) {
        istream.setstate(std::istream::failbit);
        throw ExecException(ExecError::IllegalArg);
    }

    arg = value;
}

#endif
