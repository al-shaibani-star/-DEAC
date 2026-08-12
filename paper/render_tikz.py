# -*- coding: utf-8 -*-
"""
Render each TikZ picture in DEAC_paper.tex to a standalone PNG, then write a
docx-friendly copy of the .tex where every tikzpicture is replaced by an
\includegraphics of the rendered PNG. Pandoc can then convert that copy to
.docx with all figures embedded and equations as native Word (OMML) objects.
"""
import os
import re
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "DEAC_paper.tex")
OUTTEX = os.path.join(HERE, "DEAC_paper_docx.tex")
IMGDIR = os.path.join(HERE, "tikz_img")
os.makedirs(IMGDIR, exist_ok=True)

text = open(SRC, encoding="utf-8").read()

# --- preamble (between \documentclass line and \begin{document}) ---
pre_start = text.index("\n") + 1  # skip \documentclass line
pre_end = text.index(r"\begin{document}")
preamble = text[pre_start:pre_end]

STANDALONE_HEAD = r"""\documentclass[border=10pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amssymb}
\usepackage{tikz}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
""" + "\n".join(l for l in preamble.splitlines()
                if l.strip().startswith((r"\usetikzlibrary", r"\definecolor", r"\tikzset"))
                or l.strip().startswith(r"font=") or "/.style" in l or l.strip().startswith(("io/", "proc/")))

# The above line-filter is fragile for the multiline \tikzset; instead capture
# the whole \tikzset{...} block and all \definecolor / \usetikzlibrary lines.
def capture_blocks(pre):
    libs = re.findall(r"\\usetikzlibrary\{[^}]*\}", pre)
    cols = re.findall(r"\\definecolor\{[^}]*\}\{[^}]*\}\{[^}]*\}", pre)
    # \tikzset{ ... } balanced braces (top-level)
    tikzsets = []
    i = 0
    while True:
        j = pre.find(r"\tikzset{", i)
        if j < 0:
            break
        k = j + len(r"\tikzset{") - 1  # at the '{'
        depth = 0
        m = k
        while m < len(pre):
            if pre[m] == "{":
                depth += 1
            elif pre[m] == "}":
                depth -= 1
                if depth == 0:
                    break
            m += 1
        tikzsets.append(pre[j:m + 1])
        i = m + 1
    return libs, cols, tikzsets

libs, cols, tikzsets = capture_blocks(preamble)
HEAD = (r"""\documentclass[border=10pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amssymb}
\usepackage{tikz}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
""" + "\n".join(libs) + "\n" + "\n".join(cols) + "\n" + "\n".join(tikzsets)
    + "\n\\begin{document}\n")

# --- find tikzpicture blocks ---
blocks = []
for mt in re.finditer(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", text, re.DOTALL):
    blocks.append((mt.start(), mt.end(), mt.group(0)))
print(f"Found {len(blocks)} tikzpicture blocks")

# --- render each ---
out_text = text
for idx, (s, e, blk) in enumerate(blocks, 1):
    stem = f"tikz_{idx}"
    tex_i = os.path.join(IMGDIR, stem + ".tex")
    with open(tex_i, "w", encoding="utf-8") as f:
        f.write(HEAD + blk + "\n\\end{document}\n")
    r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                        "-output-directory", IMGDIR, tex_i],
                       capture_output=True, text=True)
    pdf_i = os.path.join(IMGDIR, stem + ".pdf")
    if not os.path.exists(pdf_i):
        print(f"  [FAIL] {stem}: pdflatex error")
        tail = r.stdout[-800:]
        print(tail)
        continue
    # PDF -> PNG (300 dpi)
    subprocess.run(["pdftoppm", "-png", "-r", "300", pdf_i,
                    os.path.join(IMGDIR, stem)], capture_output=True)
    # pdftoppm names it stem-1.png
    png = os.path.join(IMGDIR, stem + "-1.png")
    if os.path.exists(png):
        final = os.path.join(IMGDIR, stem + ".png")
        if os.path.exists(final):
            os.remove(final)
        os.rename(png, final)
        print(f"  [OK] {stem}.png")
    else:
        print(f"  [FAIL] {stem}: no png produced")

# --- replace tikzpictures in the docx copy (reverse order to keep offsets) ---
for idx in range(len(blocks), 0, -1):
    s, e, blk = blocks[idx - 1]
    repl = f"\\includegraphics[width=\\linewidth]{{tikz_img/tikz_{idx}.png}}"
    out_text = out_text[:s] + repl + out_text[e:]

with open(OUTTEX, "w", encoding="utf-8") as f:
    f.write(out_text)
print(f"Wrote {OUTTEX}")
