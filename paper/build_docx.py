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

with open(OUT, "w", encoding="utf-8") as f:
    f.writelines(expand(SRC))

n_inp = sum(1 for _ in open(OUT, encoding="utf-8") if _.strip().startswith("\\input{"))
print("wrote", OUT)
print("remaining \\input lines:", n_inp)
