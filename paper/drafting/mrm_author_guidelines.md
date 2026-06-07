# MRM (Magnetic Resonance in Medicine) — Author Guidelines

Captured 2026-06-07 from the text Patrick pasted (Wiley MRM author guidelines).
Author-relevant portions reproduced faithfully; pure administrative boilerplate
(subscriber info, executive committee, corporate members, indexing services) omitted.
Submission portal: https://authors.wiley.com/journal/MRM

## Scope / review
- International journal for original methodological NMR/EPR work for medical applications;
  methodology-oriented clinical studies accepted.
- **Single-blind** review, two or more referees. Editorial decisions are final.
- Acceptance criteria: significance, originality, clarity, quality. Work must not be
  published already (preprints OK on not-for-profit sites), not under consideration elsewhere.
- ICMJE authorship: (1) substantial contribution to conception/design or acquisition/
  analysis/interpretation; AND (2) drafting/revising critically; AND (3) final approval;
  AND (4) accountability. Contributors not meeting all four → Acknowledgments.
- Copyright assigned to ISMRM on acceptance (CTA). Standard option: no page fees, color free.
- Only **one** corresponding author (need not be first author; senior author preferred).
- Equal-contribution statements: up to three equal contributors; ≤2 equal first, ≤2 equal senior.

## Preprints
- Posting a preprint on a not-for-profit site (university, arXiv, etc.) is fine and does not
  affect the editorial decision. Don't re-post the typeset version; link preprint to the
  final published version.

## Data Availability Statement (DAS) — important for us
- Strongly encouraged; must appear in the main manuscript to appear in the final paper.
- For source code/scripts: provide a link to a public repo, **preferably via a DOI**, AND
  **the SHA-1 hash uniquely identifying the specific revision used**.
- Preferred Git hosts: GitHub, BitBucket, SourceForge. Maintain ≥5 years (preferably longer).
- Human data: state conditions/approval/guidelines under which data are public (e.g. GDPR),
  and whether subjects approved public use (required in some territories). De-identify everything.
- Example statements: https://authorservices.wiley.com/open-research/index.html

## Code Reviews (RRSG)
- Optional: RRSG downloads code, checks it installs/runs. **Request in the COVER LETTER of the
  original submission only** (cannot be done at revision). Provide a 1–2 sentence paper summary
  + code description (language etc.) and ensure the code link is in the DAS.
- For software-tool papers, a Code Review is expected (editors may request one).

## Negative results
- Supported if of value to readers. Negative results that help resolve a debate or avoid
  replicating a previously-reported positive result are of considerable interest.
  (Results merely confirming accepted principles are of little interest.)

## Manuscript types & limits
| Type | Max words (body) | Max figures + tables |
|---|---|---|
| **Research Article** (default) | **5000** | **10** |
| Rapid Communication | 3500 | 7 |
| Technical Note | 2800 | 5 |
| Review | 7500 | 15 |
| Toolbox Article | 3500 | 7 |

- Word count = body text (+ appendices) only; NOT title page, abstract, figure/table captions,
  tables, or references. Revision markings excluded.
- No fixed citation limit. Limits may be relaxed only at EIC discretion; exceeding hurts acceptance.

## Human/animal studies
- Must conform to institution/country requirements; **statement that human studies were
  conducted with IRB / analogous Ethics Board approval is required.**
- For non-proof-of-concept work with statistics on humans/animals: an estimate of minimum
  sample size for statistical significance is recommended, with stated assumptions. If the
  study establishes feasibility of novel methods, state that clearly in abstract + conclusions.

## Formatting
- 1.5 line spacing; letter (8.5×11) or A4; margins all sides; basic font (Arial/Helvetica/
  Verdana/Times New Roman) ≥11 pt. Number all pages (incl. references, tables, legends).
- MS Word (preferred), RTF, or LaTeX. LaTeX: supply source (text + captions + tables, ideally
  single file) + all compile files; minimize author-defined macros; don't build figures in
  LaTeX code; upload a locally-created PDF for review. MRM LaTeX class file available (other
  classes accepted).

## Organization
- **Title page:** title (sentence case; avoid abbreviations unless common/essential; spell out
  leading numerals), authors, affiliations, corresponding author + full contact, **body word
  count**, institution info.
- **Abstract:** ≤250 words, **passive voice, avoid first person**; structured
  (**Purpose / Methods / Results / Conclusion**) for Research Articles (or Purpose / Theory and
  Methods / Results / Conclusion if a Theory section exists). **Self-contained: no formulae,
  equations, or bibliographic citations** (citations only when absolutely necessary).
- **Keywords:** 3–6, after the abstract (think SEO).
- **Body sections:** Introduction, Methods, Results, Discussion (+ optional **Theory** after
  Introduction; detailed math can go in Theory or an Appendix — appendices discouraged but
  allowed, and counted in word count). Conclusions may merge into "Discussion and Conclusions"
  if no speculation. Use heading "Methods" (not "Materials and Methods").
  - Introduction: purpose + background; avoid bulk citations; include hypothesis if applicable.
  - Results: significant findings; not a repeat of figure captions.
  - Discussion: critically evaluate, compare with literature (agreement/variance); for a new
    method, assess performance vs alternatives; confine speculation here and label it.
  - Conclusions: what can be concluded; avoid excessive claims.
- **Subheadings:** allowed, author discretion; encouraged in long Results/Discussion. (Stephan's
  advice to us: fewer subsections, use paragraphs.)
- **Tables:** title each; Arabic numerals in order of appearance; error limits + consistent sig
  figs; footnotes a,b,c below table; list at end of text or as .doc(x); NOT as .tiff/.eps.
- **Figures:** numbered in order mentioned; **callouts spelled out ("Figure 1")**; **sub-parts
  labeled A, B, C**; legible after reduction; crop/remove irrelevant + identifying info; avoid
  thin lines (≤1 pt); color encouraged (free), color images on white background. Embed in text
  for review AND upload separate image files (.eps/.tiff/.png/.pdf); name files with figure
  number. **Put all figure captions in list format at the end of the manuscript text**; do NOT
  put a caption inside its figure file.
- **Acknowledgments:** optional, between Conclusions and References; funding + assistance;
  no typists/illustrators; equal-status notes go here.
- **Footnotes:** discouraged.

## References
- Free-form style allowed; must include authors, journal/book title, article title (if any),
  year, volume/issue, pagination, optional DOI. Consistent style, numbered, brackets
  (Chicago Manual of Style numbered citations). Publisher reformats on acceptance.
- "Submitted"/"in preparation" not acceptable. Abstracts/preprints OK (published version
  supersedes). Websites only when no other publication. ≤6 authors list all; >6 list first
  three + et al. Private communication / unpublished → cite in text, not in reference list.
- Journal abbreviations: List of Journals Indexed for MEDLINE (primary).

## Supporting Information
- For material that can't fit the main article. Reference SI files in the main text. Provide
  captions both with the SI file and at the end of the main text. Label "Figure S1", "Table S1".
  Single .doc/.pdf preferred; supporting text kept minimal. Not edited; posted as submitted.

## LLM / AIGC policy
- AI-generated content (text, figures, images via ChatGPT/LLMs) **cannot be used without
  explicit editor permission**; discouraged unless part of formal research design/methods and
  clearly described (model/tool named). LLMs cannot be authors. **Spelling/grammar check is OK.**

## ORCID
- An ORCID iD is **required for submission**.

## Cover-letter / submission checklist (originals)
- Cover Letter: state manuscript type; request Code Review (if wanted) + code description + 2-sentence
  summary + code link in DAS; state if YIA competition; disclose conflicts; mention overlapping
  prior publications/abstracts.
- Title page: title, authors, affiliations, corresponding author + address, word count. (≤2 equal
  authorship; one corresponding author.)
- Conflict-of-interest document (separate upload).
- Main manuscript: abstract (structured, ≤250); figure/table caption list at end; introduce all
  figures/tables/references in numeric order; SI caption list at end; IRB/ethics statement;
  permissions for reprinted material; DAS; upload a local PDF as "Supplementary Material for Review".
- SI: single .doc/.pdf with captions; video files separately.

## Revisions (for later)
- Minor 30 days / major 60 days. Point-by-point response numbered R1.C1, R2.C1…; annotated
  manuscript with marginal R#.C# notes + highlighting; clean manuscript as Main Manuscript;
  re-upload changed figures. Professional tone required.
