# -*- coding: utf-8 -*-
r"""Flatten \input{generated/*.tex} into a single self-contained .tex for pandoc.

Produces _deac_docx_build.tex from DEAC_paper.tex so the Word export matches the
final (corrected) manuscript exactly. Figure/table paths stay relative to paper/.

Build the Word file with:
    python build_docx.py
    pandoc _deac_docx_build.tex -o DEAC_paper.docx --from=latex
"""
import os, re

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "DEAC_paper.tex")
OUT = os.path.join(HERE, "_deac_docx_build.tex")

inp_re = re.compile(r'^\s*\\input\{([^}]+)\}\s*$')

def expand(path, depth=0):
    lines_out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = inp_re.match(line)
            if m and depth < 5:
                target = m.group(1)
                if not target.endswith(".tex"):
                    target += ".tex"
                tp = os.path.join(HERE, target)
                if os.path.exists(tp):
                    lines_out.append(f"% >>> flattened from {target}\n")
                    lines_out.extend(expand(tp, depth + 1))
                    lines_out.append(f"% <<< end {target}\n")
                    continue
            lines_out.append(line)
    return lines_out

def resolve_citations(text):
    """pandoc drops \\cite and does not number a manual thebibliography, so the
    Word export loses every in-text citation and the reference numbers. Resolve
    both here: map each \\bibitem to a sequential number, rewrite \\cite{...} to
    [n] / [n, m, ...], and replace the thebibliography with a numbered, headed
    \\section*{References} list that pandoc renders cleanly."""
    m = re.search(r"\\begin\{thebibliography\}\{[^}]*\}(.*?)\\end\{thebibliography\}",
                  text, re.S)
    if not m:
        return text
    parts = re.split(r"\\bibitem\{([^}]+)\}", m.group(1))
    keymap, refs = {}, []
    for i in range(1, len(parts), 2):
        key = parts[i].strip()
        content = re.sub(r"\s+", " ", parts[i + 1]).strip()
        keymap[key] = len(refs) + 1
        refs.append((keymap[key], content))
    newbib = ("\n\\section*{References}\n\n"
              + "\n\n".join(f"[{n}]~{c}" for n, c in refs) + "\n")
    text = text[:m.start()] + newbib + text[m.end():]

    def repl(mm):
        nums = [keymap.get(k.strip()) for k in mm.group(1).split(",")]
        return "[" + ", ".join(str(x) for x in nums if x is not None) + "]"
    return re.sub(r"\\cite\{([^}]+)\}", repl, text)


flat = resolve_citations("".join(expand(SRC)))
with open(OUT, "w", encoding="utf-8") as f:
    f.write(flat)

n_inp = sum(1 for ln in flat.splitlines() if ln.strip().startswith("\\input{"))
print("wrote", OUT)
print("remaining \\input lines:", n_inp)
