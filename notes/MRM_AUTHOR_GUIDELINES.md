# MRM Author Guidelines — reference + compliance checklist for *our* submission

**Source:** Magnetic Resonance in Medicine (Wiley/ISMRM) author guidelines, copied 2026-06-06 from the MRM site (provided by Patrick). Manuscript type for us = **Research Article** (the default). Submission portal: https://authors.wiley.com/journal/MRM (Research Exchange).

This file has two parts:
1. **§A — Tailored compliance checklist + gap analysis** for *this* manuscript (the part that actually drives our final draft).
2. **§B — Verbatim guideline excerpts** (the substantive parts; pure boilerplate — subscriber/corporate/committee info — omitted).

---

## §A. Compliance checklist + gap analysis (Research Article)

### Hard limits
| Requirement | Limit | Our status (2026-06-06) | Action |
|---|---|---|---|
| Body word count | **≤ 5000** (body + appendices only; excludes title page, abstract, captions, tables, refs) | ~5,200 (python proxy; texcount TBD) — **OVER** | Concision pass + restructure trims it; verify on Overleaf texcount |
| Figures + tables combined | **≤ 10** | 9 figs + Table 1 = **10 (at cap)** | OK — every change must preserve the cap |
| Abstract | **≤ 250 words**, structured | ~266 (proxy) — **slightly over** | Trim to ≤250 |
| Keywords | **3–6** | 7 currently in main.tex | Cut to ≤6 |

### Structure (required headings)
- Manuscript divided into **Introduction, Methods, Results, Discussion**. ✅ we have these.
- **Theory** section is *optional and allowed* after Introduction — appropriate for us (Fisher / MAP / Bayesian / ADC-functional math). ✅ keep it.
- **Conclusions** may be merged into "Discussion and Conclusions" *provided no speculation/extrapolation is in it*. → decision point (we currently have a separate short Conclusion that is redundant with Discussion; merging is allowed and helps word count).
- Subheadings allowed, at author discretion. **Stephan + Patrick want fewer subsections → paragraphs.** Not a compliance issue, but reduces words + redundancy.
- Appendices are *discouraged* but allowed if the essence is understandable without them; **appendices count toward the word budget.** (Our Cramér–Rao scaling appendix counts — keep it lean or move to SI.)

### Abstract (Research Article)
- **Structured**, passive voice, no first person, ≤250 words, self-contained (no references/equations/citations except where absolutely necessary).
- Format: **Purpose / Methods / Results / Conclusion** — OR, if a Theory section is present, **Purpose / Theory and Methods / Results / Conclusion** is also acceptable. (We have a Theory section, so either works; current draft uses Purpose/Methods/Results/Conclusion.)

### Title
- Sentence case; abbreviations avoided **unless common** (ADC, MRI, DWI are common → allowed) or essential to introduce a method. Spell out numerals at the start of a title.

### Figures
- Numbered in order of first mention; callouts spelled out ("Figure 1"); sub-parts labeled **A, B, C**. Legible after reduction; avoid ≤1-pt lines; crop dead space; remove patient-identifying info + manufacturer annotation; place color on white background. **All figure captions listed (in list format) at the end of the manuscript text file** (we currently keep captions with the floats — fine for review PDF, but for the typeset LaTeX submission MRM wants the caption list at the end; verify against the MRM class).
- Figures embedded for review **and** uploaded as individual files (.eps/.tiff/.png/.pdf) at revision.

### Tables
- Title for each table; numbered Arabic; error limits given; footnotes lettered a, b, c below the table. Tables at end of text or as .doc — **not** as .tiff/.eps.

### References
- Free-form style OK at submission (publisher reformats to Chicago numbered-bracket style on acceptance). **List all authors up to 6; if >6, list first 3 then "et al."** (we must audit references.bib for this). Numbered, in order of appearance. "Submitted"/"in preparation" not allowed; preprints/abstracts allowed (note ISMRM-abstract reuse is free but must be cited). Websites only when no publication exists.

### Data Availability Statement (strongly encouraged; we should include)
- Provide a link to the public repo (we have https://github.com/PaWeRe/spectra-estimation-dMRI), **preferably a DOI**, and **the SHA-1 hash of the exact revision** used for the paper. Maintain ≥5 years. → **Action: add a Data Availability Statement with repo link + commit SHA + DOI (e.g. mint a Zenodo DOI).**
- For human data: state the conditions/approval under which data are public (IRB/GDPR etc.).
- **Code Review option (RRSG):** can request in the *cover letter of the original submission only* — they install + run the code. Given our "make it trivially reproducible / pip-installable" goal, this is a real opportunity for credibility. Decide before submitting.

### Required statements
- **IRB / ethics statement** (human subjects) — must appear in the manuscript. → **Action: confirm we have an explicit IRB-approval sentence (Methods or end).** Currently not obviously present.
- **Conflict of interest** — disclosed in cover letter + uploaded as separate file; reproduced in paper.
- **ORCID** — required for submission (corresponding author at minimum).
- **Acknowledgments** — funding (we list NIH P41EB028741, R01CA241817); equal-contribution / contribution specs go here.

### LLM / AIGC policy (important for us)
- AI-generated **text/figures cannot be used without explicit editor permission**, and never as an author. Permissible: **spelling/grammar check only.** If LLM tools were used as part of formal research design/methods, that must be clearly described with the model/tool name.
- **Our framing (already in `project_llm_policy_mrm` memory):** figures are *code-generated from data* (not AIGC), prose is Patrick's own writing/editing. Patrick to do a self-edit pass so the prose is unambiguously author-written. Keep a defensible position; disclose only what the policy requires.

### Cover letter must state
- Manuscript type (Research Article); whether requesting a Code Review (+ 2-sentence paper summary + code language); any overlapping prior publication (cite the ISMRM abstract(s) — see assets/ISMRM-2022-abstract.pdf and the 2025 submission); recommended/opposed reviewers (optional); conflicts of interest.

### Submission package checklist (from MRM)
- [ ] Cover letter (type; Code Review y/n + summary; prior ISMRM abstract citation; COI; reviewers)
- [ ] Title page (title, authors, affiliations, corresponding author + contact, **body word count**, institutions)
- [ ] Conflict-of-interest document (separate upload)
- [ ] Main manuscript: structured ≤250-word abstract; 3–6 keywords; figure/table caption list at end; all figs/tables/refs introduced in numeric order; **IRB statement**; Data Availability Statement; Supporting-Information caption list at end
- [ ] Supporting Information as a single .doc/.pdf (figures + captions), separate from main text, labeled "Figure S1…"; videos/code in native format
- [ ] Locally-compiled PDF of the whole manuscript uploaded as "Supplementary Material for Review"
- [ ] ORCID linked

### Open compliance actions for our draft (rolls into the final-draft TODO)
1. Trim body ≤5000 (Overleaf texcount) and abstract ≤250; keywords ≤6.
2. Add **Data Availability Statement** + commit SHA (+ consider Zenodo DOI) + decide on RRSG Code Review.
3. Add explicit **IRB/ethics** sentence; confirm Langkilde-2018 cohort approval covers reuse.
4. Audit `references.bib` for the ≤6-authors-then-et-al rule; ensure ISMRM abstract(s) cited.
5. Confirm **ORCID** for corresponding author; finalize corresponding-author + affiliation block (Patrick currently "independent researcher, Switzerland").
6. LLM-disclosure stance locked (code-generated figures; author-written prose).
7. Figure-caption-list-at-end + individual figure files for the LaTeX submission package.
8. Decide Conclusion: keep separate vs merge into "Discussion and Conclusions."

---

## §B. Verbatim guideline excerpts (substantive parts)

### Manuscript types & limits
- **Research Article** is the default type. (Rapid Communication 3500 words / 7 items; Technical Note 2800 / 5; Review 7500 / 15.)
- Research Article: **Max 5000 body words, max 10 figures+tables.** Word count = body of text + appendices only; **not** title page, abstract, figure captions, tables, table captions, or references. Supporting information not counted but keep it minimal. Limits may be relaxed only in exceptional cases at the Editor-in-Chief's discretion; exceeding limits decreases acceptance chance. **No fixed limit on citations.**

### Organization (verbatim intent)
- Sections: **Introduction, Methods (not "Materials and Methods"), Results, Discussion.** Optional **Theory** after Introduction for detailed math (or an Appendix; appendices discouraged, counted in words). **Conclusions** may merge with Discussion as "Discussion and Conclusions" only if no speculation/extrapolation.
- Introduction: state purpose + background; avoid "bulk" citations (weave explanatory text); include hypothesis if hypothesis-driven.
- Results: describe significant findings; **should not repeat the figure captions.**
- Discussion: critically evaluate, interpret, place in context of literature; state agreement/variance with prior work; if a new method, critically assess its performance vs alternatives. **Speculation/extrapolation minimized, confined to Discussion, clearly flagged.**
- Subheadings allowed; encouraged in long Results/Discussion.

### Abstract
- ≤250 words, structured (Purpose/Methods/Results/Conclusion **or** Purpose/Theory and Methods/Results/Conclusion), passive voice, no first person, self-contained (no formulae/equations/citations unless absolutely necessary). 3–6 keywords after the abstract (SEO matters).

### Statistical-method expectation
- For non-proof-of-concept work with statistical conclusions in humans/animals, **a minimum-sample-size estimate with stated assumptions is recommended**, and a sample consistent with it should be used. Conclusions must be supported by results. **If the purpose is feasibility of novel methods, state that clearly in the abstract and conclusions.** *(Relevant to us: the GGG n=29 grading is explicitly feasibility/exploratory; the pixel-wise demo is feasibility. Flag as such.)*

### Negative results
- Supported if of value to readership. Negative results that merely confirm accepted principles are of little interest; those that resolve a debate or save replication effort are of interest. *(Relevant: "spectrum ≈ ADC" must NOT read as a null result confirming the obvious — frame as the mechanistic explanation of why ADC works + the added uncertainty/grading contributions.)*

### Data Availability & Code Review
- Encouraged: link (preferably DOI) to public repo + **SHA-1 hash** of the revision; maintain ≥5 years. Preferred Git hosts: GitHub/BitBucket/SourceForge. Image/k-space sharing via XNAT; de-identify; IRB compliance.
- **Code Review (RRSG):** request in the **original-submission cover letter only** (cannot be done at revision); RRSG downloads + checks install/run. Software-tool-centric papers are expected to request one.

### Figures / Tables / References
- Figures: numbered by first mention; "Figure 1" spelled out; sub-parts A,B,C; legible after reduction; no ≤1-pt lines; crop; de-identify; color on white. Caption list at end of text (captions not inside the figure files). Name files with the figure number.
- Tables: titled, Arabic-numbered, error limits, lettered footnotes below; at end of text or as .doc (not tiff/eps).
- References: free-form at submission; **all authors ≤6 else first-3-et-al**; numbered in order; no "submitted/in preparation"; preprints/abstracts allowed (cite); reformatted to Chicago on acceptance.

### Formatting (for the review PDF)
- 1.5 line spacing, letter/A4, ≥11-pt basic font, margins all sides, **all pages numbered including refs/tables/legends**, line numbers. LaTeX: an MRM class file exists (encouraged, not required); supply source + all compile files; don't build figures in LaTeX code; upload a locally-built PDF.

### LLM / AIGC
- AIGC text/figures **not permitted without explicit editor permission**; cannot be an author. Spelling/grammar check is allowed. If used in formal research design/methods, describe clearly with model/tool name.

### Logistics
- Single-blind review; 2+ referees. Preprints OK on not-for-profit sites (link to final version after publication). Revisions: 30 days (minor) / 60 days (major). Resubmission of a rejected MRM paper is by EIC invitation only; a paper rejected *elsewhere* may be submitted if previous referee comments are addressed. iThenticate plagiarism/overlap check on all submissions — **cite any overlapping prior ISMRM abstracts and mention in the cover letter.**
