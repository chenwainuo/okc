"""C/C++ comment + blank-line stripper that respects string and char literals (incl. raw strings R"d(...)d")."""
import sys, os, re
def strip(src):
    out, i, n = [], 0, len(src)
    while i < n:
        c = src[i]
        if src.startswith("//", i):
            j = src.find("\n", i)
            # a // comment ending in backslash continues: keep it simple, stop at the newline
            i = n if j < 0 else j
        elif src.startswith("/*", i):
            j = src.find("*/", i + 2)
            assert j >= 0, "unterminated block comment"
            seg = src[i:j + 2]
            out.append("\n" * seg.count("\n") if "\n" in seg else " ")
            i = j + 2
        elif c == 'R' and src.startswith('R"', i) and (i == 0 or not (src[i-1].isalnum() or src[i-1] == '_')):
            m = re.match(r'R"([^(\s]*)\(', src[i:])
            if m:
                end = ")" + m.group(1) + '"'
                j = src.find(end, i + m.end())
                out.append(src[i:j + len(end)]); i = j + len(end)
            else:
                out.append(c); i += 1
        elif c == '"' or (c == "'" and not (i > 0 and (src[i-1].isalnum()))):
            q, j = c, i + 1
            while j < n and src[j] != q:
                j += 2 if src[j] == "\\" else 1
            out.append(src[i:j + 1]); i = j + 1
        else:
            out.append(c); i += 1
    return "\n".join(l.rstrip() for l in "".join(out).split("\n"))  # line numbers kept (__LINE__ feeds SASS constants)
src_dir, dst_dir = sys.argv[1], sys.argv[2]
for root, _, files in os.walk(src_dir):
    for f in files:
        p = os.path.join(root, f); q = os.path.join(dst_dir, os.path.relpath(p, src_dir))
        os.makedirs(os.path.dirname(q), exist_ok=True)
        data = open(p, encoding="utf-8", errors="surrogateescape").read()
        open(q, "w", encoding="utf-8", errors="surrogateescape").write(data if f.startswith("LICENSE") else strip(data))
