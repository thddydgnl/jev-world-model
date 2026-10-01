#!/usr/bin/env python3
"""K7: the KIIS manuscript, assembled from its text and slots.json
(kiis2026f/실험계획.md K7, kiis2026f/논문구성안.md).

    python3 scripts/kiis_paper.py          # 원고.md, figure PNGs, KIIS2026f_원고.docx
    python3 scripts/kiis_paper.py --pdf    # also render with Microsoft Word (macOS) and print the page count

The text is kiis2026f/paper/원고.tmpl.md. Every result number in it is a
{{name}} filled from kiis2026f/results/slots.json (values() below), so none is
typed by hand; an unknown name stops the build, and so does a verdict in
slots.json that the text no longer matches. The docx is assembled on the
society's Word template: its title table, styles, section breaks, headers and
page setup are kept, and its example paragraphs are the models for ours.
Figures are rendered from SVG with Quick Look (qlmanage) and cropped with
Pillow, so the build needs macOS and a Python that has Pillow. It does not
need the project venv.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "kiis2026f/paper"
RESULTS = ROOT / "kiis2026f/results"
SOURCE = PAPER / "원고.tmpl.md"
TEMPLATE = PAPER / "KIIS2026f학술대회투고양식-word.docx"
DOCX = PAPER / "KIIS2026f_원고.docx"
FIGURES = {"fig1": PAPER / "fig1.svg", "fig2": RESULTS / "fig2.svg"}

FIG_EMU = 2_860_000         # figure width, just under the column (914400 EMU per inch)
TABLE_WIDTHS = (740, 1540, 800, 1400)
MJ, GD = "신명조", "HY중고딕"

# What the text asserts; the build stops if slots.json says otherwise.
EXPECTED = {"H1": "holds", "H1b": "holds", "H2": "holds", "H3": "holds", "H5": "not distinguishable",
            "H4 k-step (test)": "holds", "H5 k-step": "holds", "H7": "holds"}


# ------------------------------------------------------------------ numbers

def values(s: dict) -> dict[str, str]:
    for h, want in EXPECTED.items():
        if s["verdicts"][h] != want:
            raise SystemExit(f"the text assumes {h} = {want}; slots.json says {s['verdicts'][h]}")
    S = s["slots"]
    r8, r15 = S["R8"], S["R15"]
    a_cfm, b_cfm, d_cfm = (r15[k] for k in ("H1b A - C_fm", "H1b B - C_fm", "H1b D - C_fm"))
    if (b_cfm["text"], b_cfm["ci_text"]) != (d_cfm["text"], d_cfm["ci_text"]) or S["R5"]["text"] != S["R6"]["text"]:
        raise SystemExit("the text reports B and D together; slots.json no longer has them equal")
    if r8["B_typed - oracle"]["positive"] or r8["B_typed - oracle"]["negative"] \
            or r8["D_gen - oracle"]["positive"] or r8["D_gen - oracle"]["negative"]:
        raise SystemExit("the text says B and D match the oracle in every episode; slots.json disagrees")
    if r8["A_jev - oracle"]["verdict"] != "not distinguishable":
        raise SystemExit("the text says JEV is not distinguishable from the oracle; slots.json disagrees")
    x1 = S["R16"]["x1_k4"]
    if not (float(x1["B_typed"]) < float(x1["A_jev"]) and float(x1["D_gen"]) < float(x1["A_jev"])):
        raise SystemExit("the text says B and D fall below JEV on the new structure; slots.json disagrees")
    cur4 = {m: S["R11"]["curves"][m]["4"]["value"] for m in ("B0_typed", "D0_gen", "persistence")}
    if not (cur4["B0_typed"] < cur4["persistence"] and cur4["D0_gen"] < cur4["persistence"]):
        raise SystemExit("the text says the untrained LLM is below persistence at four steps; slots.json disagrees")
    if r8["D0 - C_fm"]["verdict"] != "below 0":
        raise SystemExit("the text says D0 is below C_fm; slots.json disagrees")
    parsed = S["R12"]["content_on_parsed_rows"]["test"]["d0_parsed"]
    if parsed["D0_gen"] < parsed["B0_typed"]:
        raise SystemExit("the text says D0 is no less accurate than B0 on the replies that parse; slots.json disagrees")
    if S["R11"]["k_step_tests"]["D - A"]["verdict"] != "above 0":
        raise SystemExit("the text says the fine-tuned D predicts more accurately than JEV; slots.json disagrees")
    if any(S["R16"]["vs_A_jev"][k]["verdict"] != "holds" for k in ("B_typed - A_jev", "D_gen - A_jev")):
        raise SystemExit("the text says B and D drop more than JEV on the new structure; slots.json disagrees")
    mag = lambda d: d["text"].lstrip("+-")
    tf = S["R12"]["teacher_forced"]["test"]
    ent = S["R19"]["entered"]
    h7 = S["R16"]["vs_A_jev"]
    cur = S["R11"]["curves"]
    fails = S["R7"]["oracle_failures"]
    v1 = S["R17"]["v1"]["success"]
    if len({tuple(v1[a]) for a in ("A_jev", "B_typed", "D_gen", "oracle")}) != 1:
        raise SystemExit("the text says every world model matched the oracle in the first design")
    pct = lambda pair: f"{100 * pair[0] / pair[1]:.0f}"
    return {
        "C": S["R1"]["text"], "C_fm": r15["C_fm"]["text"], "A": S["R2"]["text"], "B0": S["R3"]["text"],
        "D0": S["R4"]["text"], "B": S["R5"]["text"], "D": S["R6"]["text"], "oracle": S["R7"]["oracle"]["text"],
        "x1_A_k4": x1["A_jev"], "x1_B_k4": x1["B_typed"], "x1_D_k4": x1["D_gen"],
        "n_ep": str(S["R1"]["n"]), "N": f"{S['R14']['n_transitions']:,}",
        "dA_Cfm": mag(a_cfm), "dA_Cfm_ci": a_cfm["ci_text"], "dBD_Cfm": mag(b_cfm), "dBD_Cfm_ci": b_cfm["ci_text"],
        "dB0_D0": mag(r8["H2  B0 - D0"]), "dB0_D0_ci": r8["H2  B0 - D0"]["ci_text"],
        "dD0_Cfm": mag(r8["D0 - C_fm"]), "dA_B0": r8["H5  A - B0"]["text"],
        "oracle_fail": str(sum(fails.values())), "oracle_cov": str(fails["coverage"]),
        "oracle_plan": str(fails["planner"]),
        "pf_D0": tf["D0_gen"]["text"], "pf_D": tf["D_gen"]["text"],
        "ent_D0": pct(ent["D0_gen"]), "ent_Cfm": pct(ent["C_fm"]),
        "gpu_B": f"{S['R14']['gpu_hours']['B_typed']:.1f}", "gpu_D": f"{S['R14']['gpu_hours']['D_gen']:.1f}",
        "starts": str(S["R11"]["rollouts"] // 2), "rollouts": str(S["R11"]["rollouts"]),
        "B_k4": cur["B_typed"]["4"]["text"], "D_k4": cur["D_gen"]["4"]["text"],
        "A_k1": cur["A_jev"]["1"]["text"], "A_k4": cur["A_jev"]["4"]["text"],
        "B0_k4": cur["B0_typed"]["4"]["text"], "D0_k4": cur["D0_gen"]["4"]["text"],
        "pers_k4": cur["persistence"]["4"]["text"],
        "h7_B": mag(h7["B_typed - A_jev"]), "h7_B_ci": h7["B_typed - A_jev"]["ci_text"],
        "h7_D": mag(h7["D_gen - A_jev"]), "h7_D_ci": h7["D_gen - A_jev"]["ci_text"],
        "overlap": S["R20"]["text"], "v1": f"{100 * v1['oracle'][0] / v1['oracle'][1]:.1f}",
        "r13_pass": str(S["R13"]["recursive"][0]), "r13_n": str(S["R13"]["recursive"][1]),
        "r13_direct": str(S["R13"]["direct_endpoint"][0]),
    }


def fill(text: str, vals: dict[str, str], used: set) -> str:
    def sub(m):
        name = m.group(1)
        if name not in vals:
            raise SystemExit(f"unknown name {{{{{name}}}}} in {SOURCE.name}")
        used.add(name)
        return vals[name]
    return re.sub(r"\{\{(\w+)\}\}", sub, text)


# ------------------------------------------------------------------ source

META = ("title", "title_en", "authors", "authors_en", "affil", "keywords", "abstract")


def parse(text: str) -> tuple[dict, list]:
    meta: dict = {"affil": []}
    blocks: list = []
    para: list[str] = []
    cont = None                 # meta key still being continued
    refs = False

    def flush():
        if para:
            blocks.append(("p", " ".join(para)))
            para.clear()

    for line in text.splitlines():
        line = line.rstrip()
        m = re.match(r"@(\w+):\s*(.*)", line)
        if m and m.group(1) in META:
            flush()
            key, val = m.groups()
            if key == "affil":
                meta["affil"].append(val)
                cont = None
            else:
                meta[key] = val
                cont = key
        elif not line.strip():
            flush()
            cont = None
        elif cont:
            meta[cont] += " " + line.strip()
        elif line.startswith("## "):
            flush(); blocks.append(("h2", line[3:].strip()))
        elif line.startswith("# "):
            flush(); blocks.append(("h1", line[2:].strip()))
        elif line.startswith("@figure "):
            flush(); blocks.append(("fig", [x.strip() for x in line[8:].split("|")]))
        elif line.startswith("@table "):
            flush(); blocks.append(("table", [x.strip() for x in line[7:].split("|")]))
        elif line.strip() == "@references":
            flush(); refs = True
        elif refs:
            blocks.append(("ref", line.strip()))
        else:
            para.append(line.strip())
    flush()
    return meta, blocks


def table_rows() -> list[list[str]]:
    with open(RESULTS / "table1.csv", newline="") as f:
        return list(csv.reader(f))


def markdown(meta: dict, blocks: list) -> str:
    md = lambda t: re.sub(r"\^([^^]+)\^", r"<sup>\1</sup>", re.sub(r"~([^~]+)~", r"<sub>\1</sub>", t))
    title, title_en = (meta[k].replace("<br>", " ").replace("  ", " ") for k in ("title", "title_en"))
    out = [f"# {title}", "", f"**{title_en}**", "", md(meta["authors"]), "", meta["authors_en"], ""]
    out += [md(a.replace(" | ", " · ")) for a in meta["affil"]]
    out += ["", "## 요 약", "", md(meta["abstract"]), "", f"**키워드:** {meta['keywords']}", ""]
    ref_head = False
    for kind, val in blocks:
        if kind == "h1":
            out += [f"## {val}", ""]
        elif kind == "h2":
            out += [f"### {val}", ""]
        elif kind == "p":
            out += [md(val), ""]
        elif kind == "fig":
            name, ko, en = val
            out += [f"![{ko}]({name}.png)", "", ko + "  ", en, ""]
        elif kind == "table":
            _, ko, en, note = val
            rows = table_rows()
            out += [ko + "  ", en, "", "| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(rows[0])]
            out += ["| " + " | ".join(r) + " |" for r in rows[1:]]
            out += ["", note, ""]
        elif kind == "ref":
            if not ref_head:
                out += ["## 참 고 문 헌", ""]
                ref_head = True
            out += [val + "  "]
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------ docx

SP = '<w:spacing w:line="240" w:lineRule="auto"/>'


def rfonts(font: str, hint: bool = True) -> str:
    return f'<w:rFonts w:ascii="{font}" w:eastAsia="{font}"' + (' w:hint="eastAsia"' if hint else "") + "/>"


def rpr(fonts: str = "", bold=False, italic=False, sz=None, szcs=None, va=None) -> str:
    """Run properties in schema order."""
    return ("<w:rPr>" + fonts + ("<w:b/><w:bCs/>" if bold else "") + ("<w:i/><w:iCs/>" if italic else "")
            + (f'<w:sz w:val="{sz}"/>' if sz else "") + (f'<w:szCs w:val="{szcs}"/>' if szcs else "")
            + (f'<w:vertAlign w:val="{va}"/>' if va else "") + "</w:rPr>")


def runs(text: str, fonts: str = "", **kw) -> str:
    """Inline markup (~sub~, ^sup^, *italic*, **bold**) to runs."""
    out = []
    for tok in re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*|~[^~]+~|\^[^^]+\^)", text):
        if not tok:
            continue
        extra = dict(kw)
        if len(tok) > 4 and tok.startswith("**") and tok.endswith("**"):
            extra["bold"] = True
            tok = tok[2:-2]
        elif len(tok) > 2 and tok[0] == tok[-1] and tok[0] in "*~^":
            extra.update({"*": {"italic": True}, "~": {"va": "subscript"}, "^": {"va": "superscript"}}[tok[0]])
            tok = tok[1:-1]
        out.append(f'<w:r>{rpr(fonts, **extra)}<w:t xml:space="preserve">{escape(tok)}</w:t></w:r>')
    return "".join(out)


def lines(text: str, fonts: str = "", **kw) -> str:
    """Runs with a line break wherever the text has <br> (titles broken at the colon)."""
    br = f"<w:r>{rpr(fonts, **kw)}<w:br/></w:r>"
    return br.join(runs(part.strip(), fonts, **kw) for part in text.split("<br>"))


def p(ppr: str, content: str = "") -> str:
    return f"<w:p><w:pPr>{ppr}</w:pPr>{content}</w:p>"


def blank(style: str = "a5", keep: bool = False) -> str:
    return p(f'<w:pStyle w:val="{style}"/>{"<w:keepNext/>" if keep else ""}<w:wordWrap/>{SP}')


def body(text: str) -> str:
    return p(f'<w:pStyle w:val="a5"/><w:wordWrap/>{SP}<w:ind w:firstLine="198"/>{rpr(rfonts(MJ, False))}',
             runs(text, rfonts(MJ)))


def h1(text: str) -> str:
    return p(f'<w:pStyle w:val="1"/><w:keepNext/>{SP}{rpr(rfonts(GD, False), szcs=22)}', runs(text, rfonts(GD), szcs=22))


def h2(text: str) -> str:
    # keepNext: a heading never stays alone at the foot of a column
    return p(f'<w:pStyle w:val="a5"/><w:keepNext/><w:wordWrap/>{SP}<w:ind w:firstLine="0"/>'
             + rpr(rfonts(MJ, False), bold=True), runs(text, rfonts(MJ), bold=True))


def centered(text: str, style: str = "a5", keep: bool = False) -> str:
    return p(f'<w:pStyle w:val="{style}"/>{"<w:keepNext/>" if keep else ""}<w:wordWrap/>{SP}<w:jc w:val="center"/>'
             + rpr(rfonts(MJ, False)), runs(text, rfonts(MJ)))


def picture(rid: str, name: str, cx: int, cy: int, uid: int) -> str:
    a = 'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"'
    return p(
        f'<w:pStyle w:val="a5"/><w:wordWrap/>{SP}<w:jc w:val="center"/>',
        '<w:r><w:rPr><w:noProof/></w:rPr><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0">'
        f'<wp:extent cx="{cx}" cy="{cy}"/><wp:effectExtent l="0" t="0" r="0" b="0"/>'
        f'<wp:docPr id="{uid}" name="{name}"/><wp:cNvGraphicFramePr><a:graphicFrameLocks {a} noChangeAspect="1"/>'
        f'</wp:cNvGraphicFramePr><a:graphic {a}>'
        '<a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
        '<pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:nvPicPr>'
        f'<pic:cNvPr id="{uid}" name="{name}"/><pic:cNvPicPr><a:picLocks noChangeAspect="1"/></pic:cNvPicPr>'
        f'</pic:nvPicPr><pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
        f'<pic:spPr bwMode="auto"><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
        '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/></pic:spPr></pic:pic></a:graphicData>'
        '</a:graphic></wp:inline></w:drawing></w:r>')


def table(rows: list[list[str]]) -> str:
    edge = lambda side, sz: f'<w:{side} w:val="single" w:sz="{sz}" w:space="0" w:color="000000"/>'
    out = ['<w:tbl><w:tblPr><w:tblOverlap w:val="never"/>'
           f'<w:tblW w:w="{sum(TABLE_WIDTHS)}" w:type="dxa"/><w:jc w:val="center"/><w:tblLayout w:type="fixed"/>'
           '<w:tblCellMar><w:top w:w="15" w:type="dxa"/><w:left w:w="15" w:type="dxa"/>'
           '<w:bottom w:w="15" w:type="dxa"/><w:right w:w="15" w:type="dxa"/></w:tblCellMar>'
           '<w:tblLook w:val="04A0" w:firstRow="1" w:lastRow="0" w:firstColumn="1" w:lastColumn="0" '
           'w:noHBand="0" w:noVBand="1"/></w:tblPr><w:tblGrid>'
           + "".join(f'<w:gridCol w:w="{w}"/>' for w in TABLE_WIDTHS) + "</w:tblGrid>"]
    for i, row in enumerate(rows):
        # rows do not split, and every row but the last stays with the next: the table moves as one block
        keep = "<w:keepNext/>" if i < len(rows) - 1 else ""
        out.append('<w:tr><w:trPr><w:cantSplit/><w:trHeight w:val="200"/></w:trPr>')
        for j, (text, width) in enumerate(zip(row, TABLE_WIDTHS)):
            if i == 0:
                text = text.replace(" [95% CI]", "")     # the note under the table says it
            borders = (edge("top", 12 if i == 0 else 2) + edge("left", 12 if j == 0 else 2)
                       + edge("bottom", 12 if i in (0, len(rows) - 1) else 2)
                       + edge("right", 12 if j == len(row) - 1 else 2))
            text = text.replace("C_fm", "C~fm~")
            out.append(
                f'<w:tc><w:tcPr><w:tcW w:w="{width}" w:type="dxa"/><w:tcBorders>{borders}</w:tcBorders>'
                '<w:tcMar><w:top w:w="12" w:type="dxa"/><w:left w:w="45" w:type="dxa"/>'
                '<w:bottom w:w="12" w:type="dxa"/><w:right w:w="45" w:type="dxa"/></w:tcMar>'
                '<w:vAlign w:val="center"/></w:tcPr>'
                + p(f'<w:pStyle w:val="a5"/>{keep}<w:wordWrap/>{SP}<w:ind w:firstLine="0"/>'
                    f'<w:jc w:val="{"left" if j == 0 and i else "center"}"/>{rpr(rfonts(MJ, False), sz=16, szcs=16)}',
                    runs(text, rfonts(MJ), sz=16, szcs=16)) + "</w:tc>")
        out.append("</w:tr>")
    return "".join(out) + "</w:tbl>"


def reference(text: str) -> str:
    return p('<w:pStyle w:val="a4"/><w:tabs><w:tab w:val="left" w:pos="284"/></w:tabs><w:wordWrap/>'
             f'{SP}<w:ind w:left="283" w:hangingChars="157" w:hanging="283"/>{rpr(rfonts(MJ, False))}',
             runs(text, rfonts(MJ)))


def title_table(tbl: str, meta: dict) -> str:
    """The template's one-column title table with our paragraphs in its cells."""
    head = tbl[:tbl.index("<w:tr ")]
    rows = re.findall(r"<w:tr[ >].*?</w:tr>", tbl, re.S)
    if len(rows) != 5:
        raise SystemExit("the template's title table no longer has five rows")

    def row(i: int, paras: list[str]) -> str:
        trpr = re.search(r"<w:trPr>.*?</w:trPr>", rows[i], re.S).group(0)
        if i == 3:      # the template leaves room for four affiliation lines; its minimum height follows ours
            trpr = re.sub(r'w:val="\d+"', f'w:val="{280 * max(2, len(paras))}"', trpr)
        tcpr = re.search(r"<w:tcPr>.*?</w:tcPr>", rows[i], re.S).group(0)
        return f"<w:tr>{trpr}<w:tc>{tcpr}{''.join(paras)}</w:tc></w:tr>"

    gl = '<w:rFonts w:hAnsi="굴림" w:cs="굴림"/>'
    mjgl = f'<w:rFonts w:ascii="{MJ}" w:eastAsia="{MJ}" w:hAnsi="굴림" w:cs="굴림" w:hint="eastAsia"/>'
    aff_ppr = f'<w:wordWrap/>{SP}<w:ind w:left="300" w:right="300"/><w:jc w:val="center"/>{rpr(gl, sz=20)}'
    affil = []
    for line in meta["affil"]:
        for part in line.split(" | "):
            affil.append(p(aff_ppr, runs(part, mjgl, sz=20)))
    abstract_ppr = (f'<w:pStyle w:val="-2"/><w:wordWrap/>{SP}<w:ind w:left="301" w:right="301" w:firstLine="181"/>'
                    + rpr(rfonts(MJ, False), sz=17, szcs=17))
    keyword_ppr = (f'<w:pStyle w:val="keyword"/><w:wordWrap/>{SP}<w:ind w:left="0" w:firstLineChars="200" '
                   f'w:firstLine="340"/>' + rpr(rfonts(MJ, False), bold=True, sz=17, szcs=17))
    return head + "".join([
        row(0, [p(f'<w:pStyle w:val="ab"/>{SP}{rpr(rfonts(GD, False))}', lines(meta["title"], rfonts(GD)))]),
        row(1, [p(f'<w:pStyle w:val="a4"/><w:wordWrap/>{SP}<w:jc w:val="center"/>'
                  + rpr(rfonts(GD, False), bold=True, sz=24), lines(meta["title_en"], rfonts(GD), bold=True, sz=24))]),
        row(2, [p(f'<w:pStyle w:val="-6"/>{SP}{rpr(rfonts(GD, False))}', runs(meta["authors"], rfonts(GD))),
                p(f'<w:pStyle w:val="-6"/>{SP}{rpr(rfonts(MJ, False))}', runs(meta["authors_en"], rfonts(MJ)))]),
        row(3, affil),
        row(4, [p(f'<w:pStyle w:val="-4"/>{SP}{rpr(rfonts(GD, False))}', runs("요 약", rfonts(GD))),
                p(abstract_ppr, runs(meta["abstract"], rfonts(MJ), sz=17, szcs=17)),
                p(f'<w:pStyle w:val="-2"/><w:wordWrap/>{SP}<w:ind w:left="0" w:firstLine="0"/>'),
                p(keyword_ppr, runs("키워드: ", rfonts(MJ), bold=True, sz=17, szcs=17)
                  + runs(meta["keywords"], rfonts(MJ), sz=17, szcs=17))]),
    ]) + "</w:tbl>"


def svg_to_png(svg: Path, png: Path, width: int = 1320) -> tuple[int, int]:
    """Quick Look renders the SVG into the top of a square thumbnail; keep that part."""
    from PIL import Image
    w, h = map(float, re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', svg.read_text()).groups())
    with tempfile.TemporaryDirectory() as d:
        subprocess.run(["qlmanage", "-t", "-s", str(width), "-o", d, str(svg)], check=True, capture_output=True)
        im = Image.open(Path(d) / (svg.name + ".png"))
        im = im.crop((0, 0, im.size[0], round(im.size[0] * h / w))).convert("RGB")
        im.save(png)
    return im.size


def build_docx(meta: dict, blocks: list, pngs: dict) -> None:
    with zipfile.ZipFile(TEMPLATE) as z:
        parts = {n: z.read(n) for n in z.namelist()}
    doc = parts["word/document.xml"].decode("utf-8")
    start = doc.index("<w:body>") + len("<w:body>")
    t_end = doc.index("</w:tbl>", start) + len("</w:tbl>")
    sect_par = re.match(r"<w:p [^>]*>.*?</w:p>", doc[t_end:], re.S).group(0)
    if "<w:sectPr" not in sect_par or not doc[start:].startswith("<w:tbl>"):
        raise SystemExit("the template does not start with the title table and its section break")
    final_sect = doc[doc.rindex("<w:sectPr"):doc.index("</w:body>")]

    out = [title_table(doc[start:t_end], meta), sect_par]
    rels, uid, first_h1, first_ref = [], 100, True, True
    for i, (kind, val) in enumerate(blocks):
        if kind == "h1":
            out += ([] if first_h1 else [blank(), blank()]) + [h1(val), blank(keep=True)]
            first_h1 = False
        elif kind == "h2":
            prev = blocks[i - 1][0] if i else ""
            out += ([] if prev == "h1" else [blank()]) + [h2(val)]
        elif kind == "p":
            out.append(body(val))
        elif kind == "fig":
            name, ko, en = val
            w, h = pngs[name][1]
            rid = f"rIdFig{len(rels) + 1}"
            rels.append((rid, f"media/{name}.png", pngs[name][0]))
            uid += 1
            out += [blank(), picture(rid, name, FIG_EMU, round(FIG_EMU * h / w), uid),
                    centered(ko, "a7"), centered(en, "a7"), blank()]
        elif kind == "table":
            _, ko, en, note = val
            out += [blank(), centered(ko, keep=True), centered(en, keep=True), table(table_rows()),
                    p(f'<w:pStyle w:val="a5"/><w:wordWrap/>{SP}<w:ind w:firstLine="0"/>'
                      + rpr(rfonts(MJ, False), sz=16, szcs=16), runs(note, rfonts(MJ), sz=16, szcs=16)),
                    blank()]
        elif kind == "ref":
            if first_ref:
                out += [blank(), blank(),
                        p(f'<w:pStyle w:val="-1"/><w:wordWrap/>{SP}<w:jc w:val="center"/>'
                          + rpr(rfonts(GD, False), bold=True, sz=22), runs("참 고 문 헌", rfonts(GD), bold=True, sz=22)),
                        blank("-1")]
                first_ref = False
            out.append(reference(val))
    parts["word/document.xml"] = (doc[:start] + "".join(out) + final_sect + "</w:body></w:document>").encode("utf-8")

    rel = parts["word/_rels/document.xml.rels"].decode("utf-8")
    rel = re.sub(r'<Relationship Id="[^"]+" Type="[^"]+/image" Target="media/image1.png"/>', "", rel)
    parts.pop("word/media/image1.png", None)
    image = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"
    for rid, target, path in rels:
        rel = rel.replace("</Relationships>", f'<Relationship Id="{rid}" Type="{image}" Target="{target}"/></Relationships>')
        parts["word/" + target] = path.read_bytes()
    parts["word/_rels/document.xml.rels"] = rel.encode("utf-8")
    with zipfile.ZipFile(DOCX, "w", zipfile.ZIP_DEFLATED) as z:
        for name in ["[Content_Types].xml"] + [n for n in parts if n != "[Content_Types].xml"]:
            z.writestr(name, parts[name])


def word_pdf(pdf: Path) -> int:
    """Render the docx with Microsoft Word and return the page count. Word is
    sandboxed, so the file is opened from its own Documents folder."""
    box = Path.home() / "Library/Containers/com.microsoft.Word/Data/Documents"
    box.mkdir(parents=True, exist_ok=True)
    src, dst = box / "kiis_build.docx", box / "kiis_build.pdf"
    shutil.copy(DOCX, src)
    dst.unlink(missing_ok=True)
    script = f'''
        tell application "Microsoft Word"
            open POSIX file "{src}"
            set d to active document
            save as d file name "{dst}" file format format PDF
            close d saving no
        end tell'''
    subprocess.run(["osascript", "-e", script], check=True, timeout=180, capture_output=True)
    shutil.copy(dst, pdf)
    src.unlink(missing_ok=True)
    dst.unlink(missing_ok=True)
    return len(re.findall(rb"/Type\s*/Page(?![s\w])", pdf.read_bytes()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", action="store_true", help="render with Microsoft Word and report the page count")
    args = ap.parse_args()
    slots = json.loads((RESULTS / "slots.json").read_text())
    vals, used = values(slots), set()
    source = re.sub(r"<!--.*?-->", "", SOURCE.read_text(encoding="utf-8"), flags=re.S)
    meta, blocks = parse(fill(source, vals, used))
    unused = sorted(set(vals) - used)
    if unused:
        print("names not used in the text:", ", ".join(unused))
    (PAPER / "원고.md").write_text(markdown(meta, blocks), encoding="utf-8")
    pngs = {}
    for name, svg in FIGURES.items():
        png = PAPER / f"{name}.png"
        pngs[name] = (png, svg_to_png(svg, png))
    build_docx(meta, blocks, pngs)
    chars = sum(len(v) for k, v in blocks if k == "p")
    print(f"wrote {DOCX.relative_to(ROOT)} and 원고.md  (body {chars} characters, abstract {len(meta['abstract'])})")
    if args.pdf:
        pdf = PAPER / "KIIS2026f_원고.pdf"
        print(f"Word renders {word_pdf(pdf)} page(s) -> {pdf.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
