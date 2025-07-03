def split_and_clean_lines(s: str):
    lines = s.splitlines()
    lines = map(str.strip, lines)
    lines = filter(bool, lines)
    lines = tuple(lines)
    return lines
