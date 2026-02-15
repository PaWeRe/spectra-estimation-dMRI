# Session State — MRM Paper Collaboration

> **READ THIS FIRST** when starting a new session.
> This file is the handoff document between collaborative sessions.
> Updated: 2026-02-14 (Session 1)

---

## Quick Context

We are writing an MRM journal paper on **Bayesian spectral decomposition of multi-b diffusion MRI for prostate cancer characterization**. The human (Patrick) provides high-level direction; the AI handles all implementation. See `paper/PAPER_PLAN.md` for full plan.

**Branch:** `paper/mrm-manuscript`
**Key directories:** `paper/` (LaTeX), `scripts/` (analysis), `results/` (outputs), `8640-sl6-bin/` (pixel data, gitignored)

---

## Session 1 Summary (2026-02-14)

### What we accomplished:
1. **Agentic system setup**: Created 6 skills in `.cursor/skills/`, 4 subagents in `.cursor/agents/`, and architecture docs
2. **Paper scaffold**: LaTeX skeleton in `paper/`, BibTeX refs, Makefile, PAPER_PLAN.md
3. **Data discovery**: Analyzed `8640-sl6-bin/` — confirmed **46 files = 1 b=0 + 15 b-values × 3 gradient directions** at 256×256 (native 64×64)
4. **Pixel-wise heatmap** (`scripts/pixel_wise_heatmap.py`):
   - Vectorized batch MAP estimation, 1719 pixels in <1 second
   - 8 spectral component maps + aggregate biomarker heatmap
   - Results in `results/pixel_heatmaps/`
5. **Direction independence** (`scripts/direction_comparison.py`):
   - Per-direction spectra comparison at 12 pixels across SNR range
   - D=0.25 (tumor marker) has best direction consistency
   - High-SNR pixels show near-perfect independence
   - Results in `results/direction_comparison/`
6. **Robustness test** (`scripts/robustness_test.py`):
   - 6 synthetic spectral shapes × SNR levels, NUTS recovery
   - Fast pass (2 SNR, 2 realizations) complete — all shapes converge
   - Full paper run still pending (~30 min)
   - Results in `results/robustness_test/`
7. **Gmail MCP**: Configured in `.cursor/mcp.json` (gitignored). OAuth not yet authorized — needs Cursor restart + browser authorization. **Need Patrick's email address to update config.**
8. **Paper storyline** (agreed in discussion):
   - NOT "we beat ADC by 5%"
   - YES "Bayesian decomposition → interpretable spectra + uncertainty as novel biomarker"
   - Headline finding: posterior uncertainty correlates with Gleason grade
   - Money figure: prostate segmentation + ADC vs LR probability map vs uncertainty map

6. **Flagship figure v1** (`scripts/flagship_figure.py`):
   - 4-panel: Anatomy | ADC | LR probability | Uncertainty (Laplace)
   - LR trained on 81 PZ ROIs (27 tumor, 54 normal), applied to pixels
   - ADC from monoexponential fit, uncertainty from Laplace approximation
   - Results in `results/flagship_figure/`
   - Issue: 72% pixels flagged tumor (LR calibration needs work, mask too broad)

### What's next (Session 2 priorities):
1. **Refine flagship figure**: Proper prostate segmentation (tighter mask), recalibrate LR threshold, overlay ground truth ROI contours
2. **Full robustness run**: `uv run python scripts/robustness_test.py` (no --fast flag, ~30 min)
3. **Gmail OAuth**: Patrick restarts Cursor, authorizes Gmail, then we can read Stephan's emails
4. **LaTeX**: Install basictex: `brew install --cask basictex`
5. **Email Stephan**: Request patient demographics table, b-value mapping confirmation, additional GGG case
6. **Start writing**: Theory section (all math is in the code), Methods section
7. **NUTS per pixel for select ROIs**: Run NUTS on tumor ROI pixels for gold-standard uncertainty

### Decisions made:
- Paper storyline: uncertainty as biomarker (not just classification improvement)
- Pixel-wise approach: MAP for speed + Laplace approx for uncertainty (NUTS for select ROIs)
- Mobile communication: Use Cursor cloud agents via browser (simplest approach)
- B-values: Currently using equally-spaced approximation for 16 groups; exact mapping pending from Stephan

### Open questions:
- Patrick's Gmail address (for MCP config)
- Exact b-value mapping for the 46 binary images
- Patient demographics table and additional GGG case from Stephan
- Whether to use standard segmentation model (nnU-Net) or manual contours for prostate gland

### LR classifier details (from flagship_figure.py):
- Trained on 81 PZ ROIs: 27 tumor, 54 normal
- Features: 8 spectral bin fractions + 1 combo feature = 9 features
- Key coefficients: D_0.25=+0.519 (tumor), D_3.00=-0.510 (normal), D_20.00=-0.605 (normal)
- Positive coefficients → restricted diffusion (tumor signal)
- Negative coefficients → high diffusion (normal tissue signal)

### Key file map:
| File | Purpose |
|------|---------|
| `scripts/pixel_wise_heatmap.py` | Pixel-wise MAP + heatmaps |
| `scripts/direction_comparison.py` | Direction independence analysis |
| `scripts/robustness_test.py` | NUTS robustness across spectra/SNR |
| `scripts/explore_pixel_data.py` | Data exploration (binary images) |
| `results/pixel_heatmaps/` | Heatmap figures + pixel_spectra.npz |
| `results/direction_comparison/` | Direction comparison figures + CSV |
| `results/robustness_test/` | Robustness figures + CSV |
| `paper/PAPER_PLAN.md` | Full paper plan with status |
| `paper/main.tex` | LaTeX manuscript |
| `.cursor/mcp.json` | Gmail MCP config (gitignored) |
| `secrets/` | Google OAuth credentials (gitignored) |
| `scripts/flagship_figure.py` | ADC vs spectral biomarker vs uncertainty |
| `results/flagship_figure/` | Flagship figure + numerical data |
| `8640-sl6-bin/` | 46 binary images (gitignored) |

---

---

## Session 2 Design Thought: "Co-Researcher" Agent Extension

Patrick wants to extend the agentic framework beyond paper WRITING to paper REVIEWING and general scientific co-research. Key ideas:

### Vision
A generalized "co-researcher" agent that can:
- **Review papers** (ours and others) with structured feedback
- **Follow conference-specific guidelines** (MRM formatting, NeurIPS criteria, etc.)
- **Use local models for privacy** (paper reviews must not leak to cloud APIs)
- **Integrate with the writing pipeline** (review → revise → re-review loop)

### Architecture considerations:
1. **Privacy layer**: Paper reviews should use LOCAL LLMs only (e.g., Ollama/llama.cpp with Mistral/Llama). Cloud APIs (Claude, GPT) get the final, submission-ready version only.
2. **Conference profiles**: Parameterized review criteria per venue (MRM: clinical relevance, reproducibility; NeurIPS: novelty, significance, clarity). Could be skill configs.
3. **Review workflow**: structured ScholarEval-style scoring (methodology, novelty, clarity, significance, etc.)

### Reference repositories to study:
1. **K-Dense-AI/claude-scientific-writer** (https://github.com/K-Dense-AI/claude-scientific-writer)
   - 796 stars, very mature
   - Has 20 skills including: `peer-review`, `research-lookup`, `venue-templates`, `citation-management`, `scholar-evaluation`, `hypothesis-generation`
   - Their `peer-review` skill does systematic manuscript evaluation with quantitative ScholarEval framework (8-dimension scoring)
   - Their `venue-templates` skill has conference-specific formatting
   - Could cook/flavor their skills into our system using skillchef
   - Python package with CLI and API: `pip install scientific-writer`

2. **gtarpenning/skillchef** (https://github.com/gtarpenning/skillchef)
   - Tool to sync remote skills with local customizations ("flavoring")
   - `uvx skillchef cook <github-url>` fetches a skill
   - `uvx skillchef flavor` adds local customizations that persist across syncs
   - Stores in `~/.skillchef/store/` with base/flavor/live layers
   - Symlinks into platform dirs (`~/.cursor/skills/`, `~/.codex/skills/`)
   - Perfect for importing K-Dense skills and customizing for our domain

### Proposed implementation for Session 2:
1. Install skillchef: `uvx skillchef init`
2. Cook relevant K-Dense skills: peer-review, research-lookup, venue-templates, citation-management
3. Flavor them with our MRM-specific context and local-model requirements
4. Create a new `co-researcher` skill that orchestrates: review → feedback → revision
5. Add Ollama/local model integration for privacy-sensitive operations
6. Define MRM conference profile with specific review criteria

### Questions to resolve:
- Which local model? (Mistral-7B, Llama-3.1-8B, DeepSeek-Coder?)
- How to handle the local model integration in Cursor? (MCP server? direct API?)
- Should the reviewer agent be adversarial (find weaknesses) or constructive (suggest improvements)?
- How to structure the review → revise loop to be efficient?

---

## How to Start a New Session

1. Read this file first
2. Check `git log --oneline -5` for latest commits
3. Check `paper/PAPER_PLAN.md` for task status
4. Ask Patrick: "Ready for Session N? Here's where we left off: [summary]. What's the priority today?"
