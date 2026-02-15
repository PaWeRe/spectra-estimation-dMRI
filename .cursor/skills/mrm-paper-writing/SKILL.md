---
name: mrm-paper-writing
description: Write and edit LaTeX manuscript sections for Magnetic Resonance in Medicine (MRM) journal. Use when drafting, revising, or formatting any part of the paper including abstract, introduction, theory, methods, results, discussion, conclusion, figures, tables, or references. Also use for BibTeX reference management and LaTeX compilation.
compatibility: Requires pdflatex or tectonic for compilation. Uses uv for Python scripts.
---

# MRM Paper Writing

## When to use this skill
Use when the task involves writing, editing, or formatting any part of the MRM manuscript. This includes:
- Drafting or revising LaTeX sections
- Adding or formatting references in BibTeX
- Creating figure/table LaTeX environments
- Checking MRM formatting compliance
- Compiling the manuscript PDF

## Paper structure

The manuscript lives in `paper/` with this layout:
```
paper/
├── main.tex            # Master document
├── references.bib      # BibTeX references
├── Makefile            # Build: make all, make quick
├── sections/
│   ├── abstract.tex
│   ├── introduction.tex
│   ├── theory.tex
│   ├── methods.tex
│   ├── results.tex
│   ├── discussion.tex
│   ├── conclusion.tex
│   ├── figures.tex
│   └── tables.tex
└── figures/            # Publication-quality figures (PDF/PNG)
```

## MRM journal requirements
- Double-spaced, 12pt font, 1-inch margins (already configured)
- Line numbers enabled for review
- Figures placed at end of document for submission
- Structured abstract: Purpose / Methods / Results / Conclusion
- Max 10 figures
- References in numbered style (unsrtnat)

## Writing style guidelines
- Scientific, precise, third-person passive voice
- Quantitative claims must cite specific numbers with uncertainty
- Use SI units consistently (μm²/ms for diffusivity, s/mm² for b-values)
- Define all acronyms on first use: "No-U-Turn Sampler (NUTS)"
- Cross-reference figures/tables: "Fig.~\ref{fig:spectra}" not "Figure 1"

## Custom LaTeX commands available
- `\todo{text}` — red TODO marker
- `\note{text}` — blue note marker
- `\stephan{text}` — orange "ask Stephan" marker
- `\umsmm` — μm²/ms units
- `\bval`, `\Dvec`, `\Rvec`, `\svec`, `\Umat` — math shortcuts

## Workflow
1. Read the current state of the section being edited
2. Read `paper/PAPER_PLAN.md` for the overall outline and status
3. Draft content following the outline
4. Mark uncertainties with `\todo{}` or `\stephan{}`
5. Compile with `cd paper && make quick` (no refs) or `make all` (with refs)

## Key references
See `paper/references.bib` for all citations. Key ones:
- `langkilde2018evaluation` — prior BWH dataset work
- `hoffman2014nuts` — NUTS sampler
- `gelman2006prior` — HalfCauchy prior
- `efron1993bootstrap` — Bootstrap methodology
- `lebihan1988separation` — IVIM / free water

## Collaboration protocol
- ALWAYS show the human an outline before drafting a full section
- Mark assumptions with `\todo{ASSUMPTION: ...}`
- Mark items needing Stephan with `\stephan{...}`
- Iterate: draft → human feedback → revise
