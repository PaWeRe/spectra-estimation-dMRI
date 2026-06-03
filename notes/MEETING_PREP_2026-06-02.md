# Meeting prep — Sandy + Stephan, 2026-06-02 (final pre-submission review)

**Goal of the meeting:** lock the last figure/framing decisions and agree the manuscript is submission-ready. Stephan's steer (2026-06-01 email): *don't over-rework figures; the main complaint was font sizes.* So this session is **converge + lock**, not re-open.

---

## A. What changed since the draft you sent (so you can speak to it)

All edits are in the repo (`paper/sections/*.tex`, `paper/figures/*`). **They are NOT yet in Overleaf — see §D.**

**Figures**
- **Fig 8 → `fig8_v3.pdf`** — trimmed to a clean 2×2: (a,b) normal-/tumor-like recovery, (c) δ stress test, (d) joint noise inference. **NUTS + tuned-MAP only; Gibbs boxes and the ESS panel removed.** No WIP `\todo`. Fonts matched to Fig 2.
- **Fig 5 → `fig5_v4.pdf`** — single-panel **grade ladder** (Normal / GGG=1 / GGG=2 / GGG≥3), legend on top, no subtitle. Replaces the two-boundary (emergence vs aggressiveness) version, which was redundant (the right panel reused the left's ROIs).
- **Fig 2 → `fig2_v2.pdf`** — the degenerate **D=20 single-component curve dropped** (it caused the AUC=0.000/0.224 red flags). Worst remaining grey curve now 0.49 (PZ)/0.42 (TZ); caption explains sub-chance = weak single feature, not anti-prediction.
- **Fig 7 → `fig_directions_v2.pdf`** — reworked to **whole-ROI**, patient **9283** (PZ normal + PZ tumor), 3 directions + trace average, MAP. New generator `scripts/fig7_directions_roi.py` (decoded Stephan's `.dat` export, validated r=0.977 vs canonical data). Aggregate per-direction CV across all 13 ROIs now in Results.
- **Fig 9** — caption set to **peripheral zone** per Stephan's 8640 answer; provisional note removed.
- **Atlas (149 ROIs)** — script is correct (149/149, 20 pages). The "not shown correctly" was a **stale PDF in Overleaf** + a "1..17" comment (now "1..20"). **Action: re-upload the current 20-page PDF.**
- **Dropped from SI:** multi-chain trace plots (unreadable) and the robustness-shapes figure (redundant with new Fig 8).
- **All captions drastically shortened** (they didn't fit); load-bearing detail moved into Results/Methods.

**Text**
- **Theory:** corrected the wrong Gibbs sentence — it claimed Gibbs is "ineffective" because the priors "are not conjugate." Now: *a coordinate-wise Gibbs sampler (truncated-normal conditionals + conjugate inverse-gamma σ²) targets the same posterior; we use NUTS for mixing efficiency on the correlated bins.*
- **Results:** Fig 5 text rewritten for the ladder; **ADC-sensitivity subsection reconciled to the honest version** (NUTS r=−0.79/−0.88, the r≈−0.98 is a λ=0.1 regularizer property — was contradicting the new Fig 4 caption); direction aggregate stats added; two forward-reference review comments Stephan flagged removed.
- **Methods:** MAP λ corrected **0.1 → 10⁻³** (was inconsistent with Results/figures); zone-stratification rationale sharpened with Stephan's "**the zonal contrast is a property of normal, not tumor, tissue**" point; pixel-wise (Fig 9) and direction-wise (Fig 7) provenance separated.
- **Abstract:** reframed to the four-pillar "why ADC works" spine. Dropped the 100–8000× CRLB claim, the `|r|>0.93` sensitivity-vector claim, and "systematic MAP underestimation of acinar lumen" (all λ-artifacts / unresolved). **Please review — abstract is the most-scrutinized text.**

---

## B. Decisions that need Sandy / Stephan (the actual agenda)

1. **Fig 8 / Gibbs (→ Sandy).** Trimmed figure drops Gibbs. The honest NUTS justification (mixing/ESS, not capability) is now in Theory. **Cut Gibbs from the paper entirely, or add a clean ESS/mixing panel to the SI?** (The SI panel needs the fair fully-joint inverse-gamma-σ² Gibbs first — small but not done.) Recommend: cut now, add to SI later only if Sandy wants it.
2. **Fig 8 clarity (→ Stephan).** You preferred the Fisher panel and didn't follow the old 6-panel version. Does the **trimmed recovery+noise figure read clearly** now? If you still want the Fisher correlation matrix in the main text, we can add it as a panel — otherwise it stays in SI.
3. **Fig 5 (→ Sandy, who championed by-GGG).** Confirm the **single grade ladder** is preferred over the two-boundary version. (Audit showed the two boundaries were redundant; the ladder keeps the GGG=1 emergence highlight Stephan liked + the grading axis.)
4. **Fig 7 (→ both).** Confirm **9283 / PZ** as the representative patient. **No patient in the tarball has a matched TZ normal+tumor pair** (9283 is the only matched same-zone pair, and it's PZ). MAP-only OK, or do you want per-direction NUTS (a fresh run)?
5. **CRLB / Fisher (→ Sandy).** The unconstrained-CRLB "100–8000× / 2–3 orders of magnitude" claim (Theory, Conclusion, Fisher caption panel b at SNR=150) is the one you flagged as suspicious. Needs the **van Trees Bayesian-CRLB reframe** (your `notes/CRLB_NOTE_FOR_SANDY.txt`). **Submit with it softened, or do the reframe first?**
6. **Framing/title (→ both).** Title stays "Why ADC Works"; uncertainty kept as the methodological engine + exploratory Fig 6, not the headline. Confirm.

---

## C. Known gaps still to write (you, post-meeting)

- **Methods λ-sweep paragraph** (§5d): Methods now says MAP λ=10⁻³ "tuned on simulated data" but the supporting λ-sweep paragraph/figure isn't written. Either add a short SI λ-sweep panel + paragraph, or soften the wording.
- **Results full 4-pillar restructure** (§5e Path A'): only the figure/text *contradictions* were reconciled today; the larger reorganization is still open.
- **Config drift:** `configs/prior/ridge.yaml` and `CLAUDE.md` still list λ=0.1; update to reflect 10⁻³ as the MAP default.

---

## D. Overleaf sync — do before the meeting

Edits are in the local repo only. To see them in Overleaf:
1. Commit + push the `paper/sections/*.tex` changes (or run your Overleaf-git sync).
2. Upload the **new figure PDFs**: `fig8_v3.pdf`, `fig5_v4.pdf`, `fig2_v2.pdf`, `fig_directions_v2.pdf`.
3. Re-upload the current **20-page** `figS1_all_roi_spectra.pdf` (fixes the atlas).
4. Old assets (`fig8_v2`, `fig5_v3`, `fig2_v1`, `fig_directions.png`, `fig_trace_simulation.pdf`, `fig_robustness.png`) are no longer referenced — safe to leave or delete.
