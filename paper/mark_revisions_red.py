# -*- coding: utf-8 -*-
"""
Produce a REVISION-MARKED copy of the IJIES manuscript in which the passages
added or rewritten to address the reviewers are coloured red. The clean
submission file (DEAC_paper_IJIES.docx) is left untouched; this writes a
separate DEAC_paper_IJIES_marked.docx.

Scope = reviewer-response changes: the reframed abstract and contributions, the
revised Evaluation Protocol, every Results and Discussion subsection (the
deployable-gate reframe + the new analyses: Recent Methods, Robustness,
sensitivity, controlled/selection ablations, TOST), the R1 fixes (B/H
definition, measured timing, P=20), the Conclusion, and the new
Conflicts of Interest / Acknowledgments statements.  Unchanged material
(Introduction, Related Work, the engine/pipeline method description, Datasets)
stays black.
"""
import re
from docx import Document
from docx.shared import RGBColor
from docx.oxml.ns import qn

SRC = "DEAC_paper_IJIES.docx"
OUT = "DEAC_paper_IJIES_marked.docx"
RED = RGBColor(0xC0, 0x00, 0x00)

# section headings whose whole body (and the heading itself) is reviewer-driven
RED_SECTIONS = {
    # H1
    "conclusion", "conflicts of interest", "acknowledgments",
    # H2 (Experiments + Discussion)
    "evaluation protocol", "main results",
    "comparison with traditional benchmarks", "comparison with deep clustering",
    "comparison with recent methods", "multi-metric comparison",
    "statistical significance", "optimization dynamics", "ablation study",
    "computational complexity", "why arbitration, not optimization?",
    "how much of the oracle is deployable?", "snn vs. knn graph construction",
    "robustness", "limitations",
}

# individual paragraphs (inside otherwise-unchanged sections) to colour red,
# matched by a distinctive substring
RED_SUBSTRINGS = [
    "Building on that observation, we propose DEAC",      # Contributions lead
    "An internal–external misalignment result",       # contribution 1
    "A quantified oracle–deployable gap",             # contribution 2
    "A competitive label-free, cluster-count",            # contribution 3
    "Full reproducibility artifacts",                     # contribution 4
    "In Fig",                                             # B/H definition (QualityGate)
    "the gate accepts",                                   # B/H definition
    "Are these constants fitted",                         # sensitivity (R2.2)
    "Sweeping the rho floor",                             # sensitivity (R2.2)
]


def norm_head(t):
    return re.sub(r"^\s*\d+\.\s*", "", t).strip().lower()


def color_runs(p, rgb=RED):
    for r in p.runs:
        r.font.color.rgb = rgb


def main():
    doc = Document(SRC)
    red_section = False
    n_red = 0
    for p in doc.paragraphs:
        sn = p.style.name
        txt = p.text.strip()
        if sn in ("Heading 1", "Heading 2"):
            red_section = norm_head(txt) in RED_SECTIONS
            if red_section:
                color_runs(p); n_red += 1
            continue
        if sn == "Heading 3":
            if red_section:
                color_runs(p); n_red += 1
            continue
        if not txt:
            continue
        # abstract: colour only the abstract text, keep the "Abstract:" label black
        if txt.startswith("Abstract:"):
            for r in p.runs:
                if r.text.strip() and not r.text.strip().startswith("Abstract"):
                    r.font.color.rgb = RED
            n_red += 1
            continue
        is_body = sn in ("Body Text", "First Paragraph", "Normal", "List Paragraph")
        if not is_body:
            continue  # leave captions / table cells alone
        if red_section or any(s in p.text for s in RED_SUBSTRINGS):
            color_runs(p); n_red += 1
    doc.save(OUT)
    print(f"saved {OUT}; coloured {n_red} paragraphs red")


if __name__ == "__main__":
    main()
