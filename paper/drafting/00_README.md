# Manuscript drafting workspace

Created 2026-06-07. Working context for finishing the MRM submission.
Canonical project state still lives in `../../PROJECT_STATE.md`; this folder is the
**drafting-specific** scratch + reference space.

## The plan (agreed 2026-06-07)

Order of work — **story/structure first, prose last**:

1. **Persist context** (this folder) — Patrick's notes, parsed inline comments, MRM guidelines. ✅
2. **Build `manuscript_blueprint.md`** — an "abstract version of the manuscript", content + structure only:
   - For every section: the main point(s) to make, which figure(s) it leans on, how it
     relates to the other sections, and where it sits in the overall order.
   - Bake in MRM requirements as structural elements (e.g. delete the inline GitHub
     paragraph → add a real Data Availability Statement; add IRB statement; etc.).
   - **Pressure-test every big claim** (uncertainty story, identifiability story, the
     2-bin-collapse framing, "ADC is near-optimal", the CRLB factor decomposition,
     detection-vs-grading axes, the pixel-wise "outside cohort" question).
3. **Then** conciseness pass (fit ≤5000 words) and actual prose generation from the blueprint.

## Files here

- `patrick_notes_manuscript_pass.md` — Patrick's raw "MRM final todos / manuscript pass" notes (verbatim).
- `inline_comments_inventory.md` — all `(@patrick…)` inline comments parsed from the .tex, grouped by theme, with file:line refs. Also holds the figure map + MRM compliance findings.
- `mrm_author_guidelines.md` — MRM (Wiley) author guidelines, author-relevant portions.
- `manuscript_blueprint.md` — THE work-in-progress structural manuscript (built in step 2).

## Hard MRM constraints to remember

- **Research Article**: ≤5000 words (body only), **≤10 figures + tables total**.
  Currently 9 figures + 1 table = **exactly 10. No room to add to main text.**
- Structured abstract (Purpose/Methods/Results/Conclusion), ≤250 words, passive voice, no formulae/citations.
- Required statements: IRB/ethics approval; Data Availability Statement (link + DOI + SHA-1 hash).
- LLM policy: AI only for spelling/grammar; prose must remain the authors'.
