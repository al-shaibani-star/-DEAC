# -*- coding: utf-8 -*-
"""
Re-skin paper/DEAC_paper.docx into the IJIES template format (A4, two-column body
with a single-column title block, Times New Roman, template front matter, numbered
bold section headings). Preserves all equations/tables/figures from the source.
Output: paper/DEAC_paper_IJIES.docx
"""
import copy
from docx import Document
from docx.shared import Pt, RGBColor, Emu
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

SRC = "paper/DEAC_paper.docx"
OUT = "paper/DEAC_paper_IJIES.docx"

TITLE = "DEAC: A Defensive Ensemble Framework for Automatic Graph-Based Clustering of High-Dimensional Data"
AUTHOR = "Hussein Ali Ismael¹*"
AFFIL = "¹Department of Computer Science, College of Science, University of Diyala, Diyala, Iraq"
EMAIL = "* Corresponding author’s Email: Al-Shaibani@uodiyala.edu.iq"
ABSTRACT = ("Clustering high-dimensional data without a known cluster count conventionally returns the "
    "partition that maximizes an internal quality score such as modularity or silhouette. We show this is "
    "unsafe: across twelve benchmarks, metaheuristically optimizing internal objectives degrades external "
    "agreement, because internal and external quality are only weakly correlated. We study this misalignment "
    "directly. Using a label-aware oracle that selects the best engine per dataset as an upper bound, we "
    "measure how much any label-free rule recovers: a three-layer QualityGate recovers about two-thirds of "
    "the oracle’s gain, and learned, stability, and consensus selectors never close the gap, indicating it "
    "is structural. The multi-representation Shared Nearest Neighbor and Leiden ensemble the gate arbitrates "
    "is itself a competitive label-free method that infers the cluster count, remaining statistically "
    "indistinguishable from strong classical and deep baselines while trailing only one recent method. We "
    "package these as DEAC and release all per-seed results and scripts.")
KEYWORDS = ("Clustering, High-dimensional data, Graph-based clustering, Differential evolution, "
    "Shared nearest neighbors, Leiden algorithm, Ensemble clustering")


def set_font(run, size=None, bold=None, italic=None):
    run.font.name = "Times New Roman"
    r = run._element.rPr
    if r is None:
        r = run._element.get_or_add_rPr()
    rf = r.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts"); r.insert(0, rf)
    for a in ("w:ascii", "w:hAnsi", "w:cs"):
        rf.set(qn(a), "Times New Roman")
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic
    run.font.color.rgb = RGBColor(0, 0, 0)  # force black (remove pandoc theme color)


def hrule(p):
    pPr = p._p.get_or_add_pPr()
    pbdr = OxmlElement("w:pBdr"); bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single"); bottom.set(qn("w:sz"), "6")
    bottom.set(qn("w:space"), "1"); bottom.set(qn("w:color"), "000000")
    pbdr.append(bottom); pPr.append(pbdr)


def make_sectPr(num_cols, continuous):
    s = OxmlElement("w:sectPr")
    if continuous:
        t = OxmlElement("w:type"); t.set(qn("w:val"), "continuous"); s.append(t)
    pgsz = OxmlElement("w:pgSz"); pgsz.set(qn("w:w"), "11906"); pgsz.set(qn("w:h"), "16838"); pgsz.set(qn("w:code"), "9")
    s.append(pgsz)
    pgmar = OxmlElement("w:pgMar")
    for k, v in [("top", "1440"), ("right", "1080"), ("bottom", "1440"), ("left", "1080"),
                 ("header", "850"), ("footer", "567"), ("gutter", "0")]:
        pgmar.set(qn("w:" + k), v)
    s.append(pgmar)
    cols = OxmlElement("w:cols"); cols.set(qn("w:num"), str(num_cols)); cols.set(qn("w:space"), "432")
    s.append(cols)
    return s


def main():
    doc = Document(SRC)

    # --- defaults: Times New Roman 11pt body (IJIES main-text spec) ---
    for sname in ("Normal", "Body Text", "First Paragraph", "No Spacing"):
        try:
            st = doc.styles[sname]; st.font.name = "Times New Roman"; st.font.size = Pt(11)
            st.element.rPr.rFonts.set(qn("w:cs"), "Times New Roman")
        except Exception:
            pass
    # NOTE: body indentation/justification is applied per-paragraph below (over
    # doc.paragraphs, which excludes table cells) -- NOT on the shared styles,
    # which table cells also use and which a first-line indent would wreck.

    # black-out colored cross-reference hyperlinks (pandoc colors them)
    try:
        hl = doc.styles["Hyperlink"]; hl.font.color.rgb = RGBColor(0, 0, 0); hl.font.underline = False
    except Exception:
        pass
    for hy in doc.element.body.iter(qn("w:hyperlink")):
        for rpr in hy.iter(qn("w:rPr")):
            for col in rpr.findall(qn("w:color")):
                col.set(qn("w:val"), "000000")

    paras = doc.paragraphs
    intro = next(p for p in paras if p.text.strip().lower() == "introduction")

    # --- build template front matter above Introduction ---
    def add(text, size, bold=False, italic=False, align=WD_ALIGN_PARAGRAPH.CENTER, rule=False):
        p = intro.insert_paragraph_before("")
        p.alignment = align
        p.paragraph_format.space_after = Pt(2)
        if text:
            run = p.add_run(text); set_font(run, size, bold, italic)
        if rule:
            hrule(p)
        return p

    add(TITLE, 14, bold=True)
    add(AUTHOR, 11, bold=True)
    add(AFFIL, 11, italic=False)       # single-author affiliation: regular, not italic
    email_p = add(EMAIL, 10.5)         # corresponding-author email: 10.5pt
    hrule(email_p)                 # line below author info (above abstract)
    ab = add("", 10, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    r = ab.add_run("Abstract: "); set_font(r, 10, bold=True)
    r = ab.add_run(ABSTRACT); set_font(r, 10)
    kw = add("", 10, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    r = kw.add_run("Keywords: "); set_font(r, 10, bold=True)
    r = kw.add_run(KEYWORDS); set_font(r, 10)
    hrule(kw)                      # line below keywords closes the title block
    # one blank line after the rule, before Introduction (template requirement);
    # it also carries the section break, so the keywords' last line is not
    # force-justified by an adjacent section break.
    rule_p = add("", 10)

    # --- delete OLD front matter (original paras 0-4: Title..Keywords) ---
    for p in list(doc.paragraphs):
        if p._p is intro._p:
            break
        if p.style.name in ("Title", "Author", "Abstract Title", "Abstract") or \
           (p.style.name == "First Paragraph" and p.text.startswith("Clustering, High-dimensional")):
            p._p.getparent().remove(p._p)

    # --- number + restyle section headings ---
    # back-matter statements (Conflicts of Interest, Author Contributions,
    # Acknowledgments, References) are first-order headings but stay UNnumbered.
    UNNUMBERED = {"conflicts of interest", "conflict of interest",
                  "author contributions", "acknowledgments", "acknowledgements",
                  "references", "code and data availability"}
    h1 = 0
    for p in doc.paragraphs:
        sn = p.style.name
        if sn == "Heading 1":
            txt = p.text
            if txt.strip().lower() in UNNUMBERED:
                p.clear(); run = p.add_run(txt); set_font(run, 12, bold=True)
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            else:
                h1 += 1
                p.clear(); run = p.add_run(f"{h1}.  {txt}"); set_font(run, 12, bold=True)
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        elif sn == "Heading 2":
            txt = p.text
            p.clear(); run = p.add_run(txt); set_font(run, 11, bold=True)
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        elif sn == "Heading 3":
            txt = p.text
            p.clear(); run = p.add_run(txt); set_font(run, 10, bold=True)
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT

    # --- body paragraphs: first-line indent 16.5pt, justified, no spacing ---
    # doc.paragraphs yields only body-flow paragraphs (never table cells), so this
    # leaves table cells, the Normal-styled front matter, captions and headings alone.
    for p in doc.paragraphs:
        if p.style.name in ("Body Text", "First Paragraph"):
            pf = p.paragraph_format
            pf.first_line_indent = Pt(16.5)
            pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            pf.space_before = Pt(0); pf.space_after = Pt(0); pf.line_spacing = 1.0

    # --- number the figure/table legends serially (reviewer requirement) ---
    # pandoc drops the "Figure N." / "Table N." prefix; the in-text references
    # already resolve to numbers, so we prefix each caption in document order to
    # match them. Figures use the "Image Caption" style, tables "Table Caption".
    fig_n = tab_n = 0
    for p in doc.paragraphs:
        sn = p.style.name
        if sn == "Image Caption":
            fig_n += 1; prefix = f"Fig. {fig_n}. "
        elif sn == "Table Caption":
            tab_n += 1; prefix = f"Table {tab_n}. "
        else:
            continue
        if not p.text.strip():
            continue
        # caption spec: 10pt Times New Roman, regular, centered
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for r in p.runs:
            set_font(r, 10)
        run = p.add_run(prefix)                 # created at end, then moved to front
        set_font(run, 10)
        p._p.remove(run._r)
        pPr_el = p._p.find(qn("w:pPr"))
        if pPr_el is not None:
            pPr_el.addnext(run._r)
        else:
            p._p.insert(0, run._r)

    # --- reference list: hanging indent so the [N] label sits at the margin ---
    in_refs = False
    for p in doc.paragraphs:
        if p.style.name == "Heading 1":
            in_refs = (p.text.strip().lower() == "references")
            continue
        if in_refs and p.text.strip():
            pf = p.paragraph_format
            pf.first_line_indent = Pt(-18)
            pf.left_indent = Pt(18)

    # --- sections: title block single-col, body two-col ---
    # final (doc) sectPr -> 2 columns
    body_sect = doc.element.body.find(qn("w:sectPr"))
    if body_sect is not None:
        body_sect.getparent().remove(body_sect)
    doc.element.body.append(make_sectPr(2, continuous=True))  # body continues on page 1 after title block
    # put a 1-col continuous sectPr in the closing-rule paragraph (ends the title-block section)
    pPr = rule_p._p.get_or_add_pPr()
    pPr.append(make_sectPr(1, continuous=True))

    # ---- fix horizontal overflow: fit oversized images + wide tables to the column ----
    # body column width = (usable - col space) / 2; usable = 11906 - 1080 - 1080 = 9746 twips
    COL_TW = (11906 - 1080 - 1080 - 432) // 2          # 4657 twips (3.234 in)
    FIT_TW = COL_TW - 60                               # small safety margin inside the column
    FIT_EMU = int(FIT_TW / 1440 * 914400)

    # (b) tables: fit EVERY table (top-level AND nested subfigure layout tables) to
    #     the column. doc.tables only yields top-level tables, so pandoc's nested
    #     6-inch two-column figure tables were never caught and spilled across both
    #     columns. We walk all <w:tbl> elements directly instead.
    def set_sz(rpr, halfpts):
        for tagn in ("w:sz", "w:szCs"):
            e = rpr.find(qn(tagn))
            if e is None:
                e = OxmlElement(tagn); rpr.append(e)
            e.set(qn("w:val"), str(halfpts))

    def fit_tbl_el(tbl, max_tw):
        tblPr = tbl.find(qn("w:tblPr"))
        if tblPr is None:
            tblPr = OxmlElement("w:tblPr"); tbl.insert(0, tblPr)
        lay = tblPr.find(qn("w:tblLayout"))
        if lay is None:
            lay = OxmlElement("w:tblLayout"); tblPr.append(lay)
        lay.set(qn("w:type"), "fixed")
        grid = tbl.find(qn("w:tblGrid"))
        if grid is None:
            return
        cols = grid.findall(qn("w:gridCol"))
        widths = [int(c.get(qn("w:w")) or 0) for c in cols]
        n = len(cols); total = sum(widths)
        if total <= 0 and n:
            widths = [max_tw // n] * n; total = sum(widths)
        if total > max_tw and total > 0:
            scale = max_tw / total
            widths = [max(240, int(w * scale)) for w in widths]
        for c, w in zip(cols, widths):
            c.set(qn("w:w"), str(w))
        tblW = tblPr.find(qn("w:tblW"))
        if tblW is None:
            tblW = OxmlElement("w:tblW"); tblPr.append(tblW)
        tblW.set(qn("w:type"), "dxa"); tblW.set(qn("w:w"), str(sum(widths)))
        # per-cell tcW by column position + gridSpan (direct child rows only)
        for tr in tbl.findall(qn("w:tr")):
            ci = 0
            for tc in tr.findall(qn("w:tc")):
                tcPr = tc.find(qn("w:tcPr"))
                if tcPr is None:
                    tcPr = OxmlElement("w:tcPr"); tc.insert(0, tcPr)
                span = 1
                gs = tcPr.find(qn("w:gridSpan"))
                if gs is not None:
                    span = int(gs.get(qn("w:val")) or 1)
                cw = sum(widths[ci:ci + span]) if ci + span <= len(widths) else (max_tw)
                tcW = tcPr.find(qn("w:tcW"))
                if tcW is None:
                    tcW = OxmlElement("w:tcW")
                    if gs is not None:
                        gs.addprevious(tcW)
                    else:
                        tcPr.insert(0, tcW)
                tcW.set(qn("w:type"), "dxa"); tcW.set(qn("w:w"), str(cw))
                ci += span
        # shrink font on dense DATA tables (not the 2-col subfigure layout tables)
        if n >= 7:
            fs = 16          # 8 pt
        elif n >= 5:
            fs = 18          # 9 pt
        else:
            fs = None
        if fs is not None:
            for r in tbl.findall(".//" + qn("w:r")):
                rpr = r.find(qn("w:rPr"))
                if rpr is None:
                    rpr = OxmlElement("w:rPr"); r.insert(0, rpr)
                set_sz(rpr, fs)

    # document order (parents before children) so a nested table's own settings win
    for tbl in doc.element.body.iter(qn("w:tbl")):
        fit_tbl_el(tbl, FIT_TW)

    # (c) container-aware image sizing — runs AFTER table widths are final.
    # Cap each image to its real container: the innermost table cell it sits in
    # (gridSpan x per-column width), else the body column. Walk up through nested
    # tables using the FIRST enclosing cell + its table.
    def container_twips(inline_el):
        el = inline_el.getparent()
        tc = None
        while el is not None:
            tag = el.tag.split("}")[1]
            if tag == "tc":
                tc = el
                break
            el = el.getparent()
        if tc is None:
            return FIT_TW
        tbl = tc.getparent()
        while tbl is not None and tbl.tag.split("}")[1] != "tbl":
            tbl = tbl.getparent()
        if tbl is None:
            return FIT_TW
        grid = tbl.find(qn("w:tblGrid"))
        gw = [int(c.get(qn("w:w")) or 0) for c in grid.findall(qn("w:gridCol"))]
        total = sum(gw)
        if total <= 0:
            return FIT_TW
        # this cell's width = its own tcW if present, else gridSpan share of the grid
        tcPr = tc.find(qn("w:tcPr"))
        tcW = tcPr.find(qn("w:tcW")) if tcPr is not None else None
        if tcW is not None and tcW.get(qn("w:w")):
            return int(tcW.get(qn("w:w")))
        span = 1
        if tcPr is not None:
            gs = tcPr.find(qn("w:gridSpan"))
            if gs is not None:
                span = int(gs.get(qn("w:val")) or 1)
        return int(total / len(gw) * span)

    for sh in doc.inline_shapes:
        avail_tw = container_twips(sh._inline) - 120      # cell/column padding margin
        avail_emu = max(int(0.4 * 914400), int(avail_tw / 1440 * 914400))
        if sh.width and sh.width > avail_emu:
            ratio = avail_emu / sh.width
            sh.height = Emu(int(sh.height * ratio))
            sh.width = Emu(avail_emu)

    doc.save(OUT)
    print("saved", OUT)


if __name__ == "__main__":
    main()
