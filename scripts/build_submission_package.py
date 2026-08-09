"""Assemble the MRM submission package from the manuscript sources.

The journal's upload form wants the Main Manuscript WITHOUT supplementary
material, so this script splits paper/ into two independently compilable
documents and stages everything else the form asks for:

  paper/submission/
    01_main_manuscript/ (+.zip)  -> "Main Manuscript" slot (LaTeX archive)
    02_supporting_information/   -> "LaTeX Supplementary File" slot
    03_figures/                  -> optional "Figure" slot, MRM-named
    04_conflict_of_interest/     -> "Conflict of Interest" slot, one per author
    05_cover_letter/             -> "Cover letter / Comments" slot

The two cross-document references are resolved to literals, since neither
document can see the other's labels:
  main manuscript : \\ref{fig:atlas|directions|convergence} -> S1 / S2 / S3
  supporting info : \\ref{eq:spacing_scaling} -> [4];  \\ref{fig:fisher} -> Figure 1

SI figure order follows order of first mention in the body (MRM rule); it is
asserted below rather than assumed, so moving a mention breaks the build
instead of silently mis-numbering.

Usage:  uv run python scripts/build_submission_package.py
"""

import json
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PAP = REPO / "paper"
OUT = PAP / "submission"

TITLE = "Why ADC works: Bayesian spectral decomposition of prostate multi-b diffusion MRI"

# Order of appearance in sections/figures.tex.
MAIN_FIGS = [
    ("fig_fisher_v2.pdf", "Figure1"), ("fig3_v7.pdf", "Figure2"),
    ("fig1_v4.pdf", "Figure3"), ("fig2_v3.pdf", "Figure4"),
    ("fig3_v4.pdf", "Figure5"), ("fig4_std_v4.pdf", "Figure6"),
    ("fig5_v5.pdf", "Figure7"), ("fig6_v2.pdf", "Figure8"),
    ("fig9_v2.pdf", "Figure9"),
]
SI_FIGS = ["figS_subset_atlas.pdf", "fig_directions_v4.pdf",
           "figS_subset_convergence.pdf"]
BODY = ["abstract", "introduction", "theory", "methods", "results",
        "discussion", "conclusion"]

AUTHORS = [
    ("Patrick Remerscheid",
     "Independent researcher, Switzerland; formerly Department of Radiology, "
     "Brigham and Women's Hospital, Harvard Medical School, Boston, MA, USA",
     "COI_Remerscheid"),
    ("William M. Wells III",
     "Department of Radiology, Brigham and Women's Hospital, Harvard Medical "
     "School, Boston, MA, USA; Computer Science and Artificial Intelligence "
     "Laboratory (CSAIL), Massachusetts Institute of Technology, Cambridge, "
     "MA, USA",
     "COI_Wells"),
    ("Stephan E. Maier",
     "Department of Radiology, Brigham and Women's Hospital, Harvard Medical "
     "School, Boston, MA, USA",
     "COI_Maier"),
]


def si_order() -> list[str]:
    """SI labels in order of first mention in the body (the MRM numbering rule)."""
    seen: list[str] = []
    for name in BODY:
        text = (PAP / "sections" / f"{name}.tex").read_text()
        for m in re.finditer(r"\\ref\{(fig:(?:atlas|convergence|directions))\}", text):
            if m.group(1) not in seen:
                seen.append(m.group(1))
    return seen


def float_order() -> list[str]:
    """SI labels in the order their floats appear in supporting.tex."""
    text = (PAP / "sections" / "supporting.tex").read_text()
    text = re.sub(r"(?m)^\s*%.*$", "", text)          # drop comments
    return re.findall(r"\\label\{(fig:[a-z]+)\}", text)


def rtf_escape(s: str) -> str:
    return (s.replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}")
             .replace("'", r"\rquote "))


def coi_rtf(author: str, affil: str) -> str:
    paras = [
        r"\b\fs32 Conflict of Interest Disclosure\b0\fs24", "",
        r"\b Manuscript:\b0  " + rtf_escape(TITLE),
        r"\b Journal:\b0  Magnetic Resonance in Medicine",
        r"\b Author:\b0  " + rtf_escape(author),
        r"\b Affiliation:\b0  " + rtf_escape(affil), "",
        r"\b Declaration\b0",
        "The author named above declares no conflict of interest in connection "
        "with this manuscript. Specifically, the author has no patents or patent "
        "applications, industry consulting arrangements, honoraria or speaking "
        "fees, equity holdings, advisory or board positions, or industry-funded "
        "research support that relate to the work described in this manuscript.",
        "",
        r"\b Research support\b0",
        "This study was supported in part by National Institutes of Health grants "
        "P41EB028741 and R01CA241817. These are non-industry, peer-reviewed public "
        "grants; they are disclosed in the Acknowledgments of the manuscript and "
        "do not constitute a conflict of interest.",
        "", "", r"Signature: \line \line Date:",
    ]
    body = "".join(r"{\pard\sa180 " + p + r"\par}" + "\n" for p in paras)
    return (r"{\rtf1\ansi\ansicpg1252\deff0"
            r"{\fonttbl{\f0\froman Times New Roman;}}"
            "\\viewkind4\\uc1\\pard\\f0\\fs24\n" + body + "}\n")


COMPILED = REPO / "paper" / "compiled"
PDF_JOBS = [
    # (source name in paper/compiled, staged name, must contain, must NOT contain)
    ("main_manuscript.pdf", "01_main_manuscript.pdf",
     ["Why ADC works", "Discussion", "Data Availability Statement"],
     ["Supporting Information"]),
    ("supporting_information.pdf", "02_supporting_information.pdf",
     ["Supporting Information", "Cram"],
     []),
]


def pdf_text(path: Path) -> str:
    return subprocess.run(["pdftotext", "-q", str(path), "-"],
                          capture_output=True, text=True).stdout


def pdf_pages(path: Path) -> int:
    out = subprocess.run(["pdfinfo", str(path)], capture_output=True, text=True).stdout
    m = re.search(r"^Pages:\s+(\d+)", out, re.M)
    return int(m.group(1)) if m else -1


def stage_pdfs(out: Path) -> None:
    """Copy hand-dropped Overleaf PDFs into the package and sanity-check them.

    Nothing here can be produced locally (no TeX), so the checks stand in for a
    build: right document, Supporting Information genuinely absent from the
    manuscript, and no unresolved LaTeX references left on the page.
    """
    if not COMPILED.exists():
        return
    dest = out / "06_compiled_pdf"
    for src_name, staged_name, must, must_not in PDF_JOBS:
        src = COMPILED / src_name
        if not src.exists():
            print(f"  [pdf] {src_name}: not provided, skipping")
            continue
        if src.read_bytes()[:5] != b"%PDF-":
            raise SystemExit(f"{src} is not a PDF")
        text = pdf_text(src)
        for needle in must:
            if needle not in text:
                raise SystemExit(f"{src_name}: expected text {needle!r} not found "
                                 "-- is this the right document?")
        for needle in must_not:
            if needle in text:
                raise SystemExit(f"{src_name}: contains {needle!r}, which must not "
                                 "appear -- did you compile the pre-split main.tex "
                                 "instead of 01_main_manuscript.zip?")
        if "??" in text:
            raise SystemExit(f"{src_name}: contains '??' -- unresolved LaTeX "
                             "reference; recompile (LaTeX needs two passes).")
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dest / staged_name)
        print(f"  [pdf] {staged_name}: {pdf_pages(src)} pages, checks passed")



MANIFEST = ".build_manifest.json"


def clean_generated(out: Path) -> None:
    """Remove only what a previous run generated, never anything else.

    An earlier version wiped paper/submission/ wholesale, which destroyed two
    PDFs a human had dropped in there by hand -- shutil.rmtree does not go to
    the Trash, so they were unrecoverable. Generated paths are now recorded in
    a manifest and only those are removed; anything unrecognised is left in
    place and reported.
    """
    if not out.exists():
        return
    known = set()
    mf = out / MANIFEST
    if mf.exists():
        try:
            known = set(json.loads(mf.read_text()))
        except (json.JSONDecodeError, OSError):
            known = set()
    present = {str(q.relative_to(out)) for q in out.rglob("*")
               if q.is_file() and not q.name.startswith(".")}
    present.discard(MANIFEST)
    strangers = sorted(present - known)
    if strangers and not known:
        raise SystemExit(
            f"{out} holds files but no build manifest, so this script cannot "
            "tell what is safe to delete. Move anything you want to keep out "
            "of this folder (hand-made files belong in paper/compiled/), then "
            "re-run.\n  " + "\n  ".join(strangers))
    for rel in known:
        f = out / rel
        if f.is_file():
            f.unlink()
    for d in sorted((q for q in out.rglob("*") if q.is_dir()),
                    key=lambda q: len(q.parts), reverse=True):
        if not any(d.iterdir()):
            d.rmdir()
    mf.unlink(missing_ok=True)
    if strangers:
        print(f"  [keep] left {len(strangers)} file(s) this script did not "
              "generate:")
        for s in strangers:
            print(f"         {s}")


def write_manifest(out: Path) -> None:
    files = sorted(str(q.relative_to(out)) for q in out.rglob("*")
                   if q.is_file() and not q.name.startswith("."))
    (out / MANIFEST).write_text(json.dumps(files, indent=1))



CHROME = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")

COI_CSS = """
@page { size: letter; margin: 1in; }
body { font-family: "Times New Roman", Times, serif; font-size: 12pt;
       line-height: 1.55; color: #000; }
h1   { font-size: 15pt; margin: 0 0 20pt; }
h2   { font-size: 12pt; margin: 20pt 0 4pt; }
table.meta { border-collapse: collapse; margin-bottom: 6pt; }
table.meta td { padding: 1pt 0; vertical-align: top; }
table.meta td.k { font-weight: bold; white-space: nowrap; padding-right: 10pt; }
p { margin: 0 0 10pt; text-align: justify; }
.sig { margin-top: 46pt; }
.sig td { padding-top: 26pt; border-bottom: 1px solid #000; }
.sig td.gap { border-bottom: none; width: 32pt; }
.sig .lbl { border-bottom: none; padding-top: 3pt; font-size: 10pt; }
"""


def coi_html(author: str, affil: str) -> str:
    """Signable one-page disclosure. Content identical to the RTF version."""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Conflict of Interest Disclosure - {esc_html(author)}</title>
<style>{COI_CSS}</style></head><body>
<h1>Conflict of Interest Disclosure</h1>
<table class="meta">
  <tr><td class="k">Manuscript</td><td>{esc_html(TITLE)}</td></tr>
  <tr><td class="k">Journal</td><td>Magnetic Resonance in Medicine</td></tr>
  <tr><td class="k">Author</td><td>{esc_html(author)}</td></tr>
  <tr><td class="k">Affiliation</td><td>{esc_html(affil)}</td></tr>
</table>

<h2>Declaration</h2>
<p>The author named above declares no conflict of interest in connection with this
manuscript. Specifically, the author has no patents or patent applications, industry
consulting arrangements, honoraria or speaking fees, equity holdings, advisory or board
positions, or industry-funded research support that relate to the work described in this
manuscript.</p>

<h2>Research support</h2>
<p>This study was supported in part by National Institutes of Health grants P41EB028741
and R01CA241817. These are non-industry, peer-reviewed public grants; they are disclosed
in the Acknowledgments of the manuscript and do not constitute a conflict of
interest.</p>

<table class="sig" width="100%">
  <tr><td width="55%">&nbsp;</td><td class="gap"></td><td>&nbsp;</td></tr>
  <tr><td class="lbl">Signature</td><td class="gap"></td><td class="lbl">Date</td></tr>
</table>
</body></html>"""


def esc_html(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
             .replace("'", "\u2019"))


def html_to_pdf(html: Path, pdf: Path, profile: str) -> bool:
    """Render with headless Chrome.

    Chrome reliably writes the PDF but does not always exit afterwards, so we
    poll for the finished file and then kill it rather than waiting on the
    process. `profile` is a throwaway user-data-dir, shared across calls, so a
    running Chrome session is untouched and first-run setup happens once.
    """
    if not CHROME.exists():
        return False
    if pdf.exists():
        pdf.unlink()
    proc = subprocess.Popen(
        [str(CHROME), "--headless=new", "--disable-gpu",
         f"--user-data-dir={profile}", "--no-pdf-header-footer",
         f"--print-to-pdf={pdf}", html.resolve().as_uri()],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    size, stable, waited = -1, 0, 0.0
    try:
        while waited < 90:
            if pdf.exists():
                now = pdf.stat().st_size
                stable = stable + 1 if now == size and now > 1000 else 0
                size = now
                if stable >= 2:            # size unchanged across two polls
                    break
            if proc.poll() is not None and waited > 1:
                break
            time.sleep(0.4)
            waited += 0.4
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    if not pdf.exists() or pdf.read_bytes()[:5] != b"%PDF-":
        raise SystemExit(f"Chrome failed to render {html.name}")
    return True


SI_PREAMBLE = r"""% ======================================================================
% Supporting Information for:
%   "Why ADC works: Bayesian spectral decomposition of prostate multi-b
%    diffusion MRI"  --  Remerscheid, Wells, Maier
% Submitted to Magnetic Resonance in Medicine.
% Generated by scripts/build_submission_package.py -- edit paper/sections/
% supporting.tex, not this file.
% ======================================================================
\documentclass[11pt]{article}
\usepackage{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage[english]{babel}
\usepackage{amsmath,amsfonts,amssymb}
\usepackage{graphicx}
\usepackage{siunitx}
\usepackage{hyperref}
\usepackage{mathtools}
\usepackage{titlesec}
\geometry{letterpaper,margin=1in}
\hypersetup{colorlinks=true,linkcolor=blue,citecolor=blue,urlcolor=blue}
\linespread{1.5}
\titleformat{name=\section}{\normalfont\Large\bfseries}{}{0pt}{}
\titleformat{name=\subsection}{\normalfont\large\bfseries}{}{0pt}{}
\newcommand{\umsmm}{\si{\micro\meter\squared\per\milli\second}}
\newcommand{\figbox}[1]{\includegraphics[width=\textwidth,height=0.95\textheight,keepaspectratio]{#1}}
\renewcommand{\thefigure}{S\arabic{figure}}

\begin{document}
{\noindent\LARGE\bf Supporting Information}
\medskip

{\noindent\large Why ADC works: Bayesian spectral decomposition of prostate
multi-b diffusion MRI}

\medskip
\noindent Patrick Remerscheid, William M.\ Wells III, Stephan E.\ Maier
\bigskip

"""


def main() -> None:
    mention, floats = si_order(), float_order()
    assert mention == floats, (
        "SI figure floats are not in order of first mention -- reorder the "
        f"floats in sections/supporting.tex.\n  mentioned: {mention}\n"
        f"  floats   : {floats}")
    si_num = {lab: f"S{i}" for i, lab in enumerate(mention, 1)}
    print("SI numbering:", ", ".join(f"{v}={k}" for k, v in si_num.items()))

    clean_generated(OUT)

    # --- 1. Main manuscript, Supporting Information removed -----------------
    mm = OUT / "01_main_manuscript"
    (mm / "sections").mkdir(parents=True)
    (mm / "figures").mkdir()
    main = (PAP / "main.tex").read_text()
    block = ("% === Supporting Information ===\n%TC:ignore\n\\clearpage\n"
             "\\renewcommand{\\thefigure}{S\\arabic{figure}}\n"
             "\\setcounter{figure}{0}\n\\input{sections/supporting}\n"
             "%TC:endignore\n\n")
    assert block in main, "Supporting Information block not found in main.tex"
    main = main.replace(block, (
        "% === Supporting Information ===\n"
        "% Submitted separately (MRM: the Main Manuscript must not contain\n"
        "% supplementary material). See 02_supporting_information/.\n\n"))
    (mm / "main.tex").write_text(main)
    shutil.copy(PAP / "references.bib", mm / "references.bib")
    for f in sorted((PAP / "sections").glob("*.tex")):
        if f.name == "supporting.tex":
            continue
        text = f.read_text()
        for lab, num in si_num.items():
            text = re.sub(r"\\ref\{" + re.escape(lab) + r"\}", num, text)
        (mm / "sections" / f.name).write_text(text)
    for src, _ in MAIN_FIGS:
        shutil.copy(PAP / "figures" / src, mm / "figures" / src)

    # --- 2. Standalone Supporting Information --------------------------------
    si = OUT / "02_supporting_information"
    (si / "figures").mkdir(parents=True)
    sup = (PAP / "sections" / "supporting.tex").read_text()
    sup = sup.replace("Equation~\\ref{eq:spacing_scaling} in the\nmain text",
                      "Equation~[4] in the\nmain text")
    sup = sup.replace("Figure~\\ref{fig:fisher}", "Figure~1 of the main text")
    sup = re.sub(r"%TC:(end)?ignore\n", "", sup)
    assert "\\ref{eq:" not in sup and "\\ref{fig:fisher}" not in sup, \
        "unresolved main-text reference left in the standalone SI"
    (si / "si.tex").write_text(SI_PREAMBLE + sup + "\n\\end{document}\n")
    for src in SI_FIGS:
        shutil.copy(PAP / "figures" / src, si / "figures" / src)

    # --- 3. Figures, separately, MRM-named -----------------------------------
    fg = OUT / "03_figures"
    fg.mkdir(parents=True)
    for src, dst in MAIN_FIGS:
        shutil.copy(PAP / "figures" / src, fg / f"{dst}.pdf")

    # --- 4. Conflict of interest, one document per author --------------------
    coi = OUT / "04_conflict_of_interest"
    coi.mkdir(parents=True)
    made = 0
    with tempfile.TemporaryDirectory() as profile:
        for author, affil, stem in AUTHORS:
            (coi / f"{stem}.rtf").write_text(coi_rtf(author, affil))
            html = coi / f"{stem}.html"
            html.write_text(coi_html(author, affil))
            try:
                if html_to_pdf(html, coi / f"{stem}.pdf", profile):
                    made += 1
            finally:
                html.unlink()
    print(f"  [coi] {len(AUTHORS)} disclosures as RTF"
          + (f" and {made} as PDF (upload the PDFs)" if made
             else " only -- Chrome unavailable, so no PDF"))

    # --- 4b. Overleaf-compiled PDFs, if they have been dropped in ------------
    stage_pdfs(OUT)

    # --- 5. Cover letter -----------------------------------------------------
    cl = OUT / "05_cover_letter"
    cl.mkdir(parents=True)
    shutil.copy(PAP / "drafting" / "cover_letter.pdf", cl / "cover_letter.pdf")

    # --- archives for the two LaTeX slots ------------------------------------
    for d in (mm, si):
        subprocess.run(["zip", "-qr", f"../{d.name}.zip", ".", "-x", ".*"],
                       cwd=d, check=True)

    # --- validation: every reference resolves inside its own document --------
    for label, files in [
        ("main manuscript",
         [mm / "main.tex"] + sorted((mm / "sections").glob("*.tex"))),
        ("supporting information", [si / "si.tex"]),
    ]:
        body = "".join(f.read_text() for f in files)
        labels = set(re.findall(r"\\label\{([^}]*)\}", body))
        refs = set(re.findall(r"\\(?:ref|eqref)\{([^}]*)\}", body))
        assert not refs - labels, f"{label}: unresolved \\ref {refs - labels}"
    bib = set(re.findall(r"@\w+\{([^,]+),",
                         (mm / "references.bib").read_text()))
    body = "".join(f.read_text() for f in sorted((mm / "sections").glob("*.tex")))
    cites = {k.strip() for m in re.findall(r"\\cite[a-z]*\{([^}]*)\}", body)
             for k in m.split(",")}
    assert not cites - bib, f"missing bib keys: {cites - bib}"
    assert not bib - cites, f"uncited bib entries: {bib - cites}"

    write_manifest(OUT)
    print(f"Wrote {OUT.relative_to(REPO)} -- see paper/drafting/SUBMISSION_PACKAGE.md")


if __name__ == "__main__":
    main()
