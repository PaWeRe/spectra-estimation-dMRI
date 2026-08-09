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

import re
import shutil
import subprocess
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

    if OUT.exists():
        shutil.rmtree(OUT)

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
    for author, affil, stem in AUTHORS:
        (coi / f"{stem}.rtf").write_text(coi_rtf(author, affil))

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

    print(f"Wrote {OUT.relative_to(REPO)} -- see paper/drafting/SUBMISSION_PACKAGE.md")


if __name__ == "__main__":
    main()
