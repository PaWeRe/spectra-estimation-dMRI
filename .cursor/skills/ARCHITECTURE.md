# Agentic System Architecture

## Overview

Human-AI collaborative system for an MRM journal paper on Bayesian spectral
decomposition of multi-b diffusion MRI. The human (Patrick) provides direction
and scientific judgment; the agent handles implementation.

## System Design

```
HUMAN (Patrick) — direction, feedback, scientific judgment
       │
       ▼
ORCHESTRATOR (main Cursor agent)
  ├── reads: SESSION.md (moving memory), skills/ (domain knowledge)
  ├── delegates to subagents for context-heavy parallel work
  └── updates SESSION.md + commits at end of each session

SUBAGENTS (.cursor/agents/) — parallel, context-isolated workers
  ┌─────────────┬─────────────┬─────────────┬──────────────┐
  │ researcher  │ computation │  verifier   │ latex-writer  │
  │ (fast,      │ (inherit,   │ (fast,      │ (inherit)     │
  │  readonly)  │  background)│  readonly)  │               │
  │ understand  │ run scripts │ validate    │ draft paper   │
  │ codebase    │ gen figures │ results     │ sections      │
  └─────────────┴─────────────┴─────────────┴──────────────┘

SKILLS (.cursor/skills/) — domain knowledge reference docs
  ├── mrm-paper-writing    — journal style, structure
  ├── spectrum-estimation  — Bayesian model, NUTS, MAP
  ├── biomarker-creation   — LR, LOOCV, AUC, uncertainty
  ├── image-processing     — DWI loading, trace computation
  ├── benchmarking-visualization — figures, plots, styling
  └── general-coding       — Python, uv, Hydra, git
```

## Shared Memory: SESSION.md

SESSION.md is the **single source of truth** across all sessions and agents.
- Every session reads it first
- Every session updates it at the end
- Every subagent reads it before starting work
- It is committed to git after each session

## When to delegate

| Task | Agent | Why |
|------|-------|-----|
| "What does the NUTS sampler do?" | researcher | Code exploration, context-heavy |
| "Run the biomarker pipeline" | computation | Long-running, background |
| "Check that AUCs match the paper" | verifier | Validation, fast model |
| "Draft the methods section" | latex-writer | Domain writing, context-isolated |
| "Fix this bug" / "Design this feature" | orchestrator (self) | Needs conversation context |

## Data Flow

```
Raw Data (.bin)  →  load_prostate_dwi()  →  trace images (15 b-values)
                                                    │
BWH ROI JSON     →  load_bwh_signal_decays()  →  signal decays (149 ROIs)
                                                    │
                                              ProbabilisticModel
                                              (design matrix U, prior)
                                                    │
                                         ┌──────────┼──────────┐
                                         MAP     NUTS      (future: NN)
                                         │         │
                                    spectrum   spectrum + σ + uncertainty
                                         │         │
                                         └────┬────┘
                                              │
                                     Biomarker Pipeline
                                     (LR, LOOCV, AUC, bootstrap)
                                              │
                                         Figures + LaTeX
```

## Patrick's Preferences (persistent across sessions)

- Use `uv` for all Python commands
- Integrate new code into existing `src/` package, don't create standalone scripts
- Think methodology first, code second
- Focus on clinical utility: why is Bayesian better than ADC?
- Keep the pipeline unified (pixel-level = generalization of ROI-level)
- MRM allows max 10 figures — every figure must earn its place
- Uncertainty quantification is a key selling point
- Questions go through Patrick, not directly to co-authors
