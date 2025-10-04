import re

def parse_ranges(ranges_text: str):
    """
    Parse ranges like: "1-50, 120-140, 200" -> list of (start, end).
    Single ints become (n, n).
    """
    spans = []
    for token in re.split(r"[,\s]+", ranges_text.strip()):
        if not token:
            continue
        if "-" in token:
            a, b = token.split("-", 1)
            a, b = int(a), int(b)
            if a > b:
                a, b = b, a
            spans.append((a, b))
        else:
            n = int(token)
            spans.append((n, n))
    return spans

def is_hex_color(s: str) -> bool:
    return bool(re.fullmatch(r"#([0-9a-fA-F]{6})", s))