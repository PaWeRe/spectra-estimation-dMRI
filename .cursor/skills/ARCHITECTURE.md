# Agentic System Architecture

## Overview

A human-AI collaborative system for producing an MRM journal manuscript.
The human provides high-level direction; the agent handles all implementation.

## System Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        HUMAN (Patrick)                              │
│  Role: Strategic direction, scientific judgment, feedback           │
│  Interface: Terminal (Cursor chat)                                  │
│  Actions: Approve plans, give feedback, contact co-authors          │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                    high-level direction
                         feedback
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR AGENT (Cursor/Claude)                │
│                                                                     │
│  Role: Task decomposition, skill activation, progress tracking      │
│  Context: .cursorrules + PAPER_PLAN.md + TODO list                  │
│                                                                     │
│  Capabilities:                                                      │
│  • Reads code, configs, data, papers                                │
│  • Writes code, LaTeX, configs                                      │
│  • Runs scripts via shell (uv run python ...)                       │
│  • Browses web for references, journal guidelines                   │
│  • Manages git (branch, commit, diff)                               │
│  • Tracks progress via TODO system                                  │
│  • Activates skills based on task type                              │
└───┬─────────┬──────────┬──────────┬──────────┬──────────┬───────────┘
    │         │          │          │          │          │
    ▼         ▼          ▼          ▼          ▼          ▼
┌────────┐┌────────┐┌──────────┐┌────────┐┌──────────┐┌────────┐
│  MRM   ││General ││ Spectrum ││Biomark-││Benchmark-││ Image  │
│ Paper  ││Coding  ││Estimation││  er    ││  ing &   ││Process-│
│Writing ││        ││          ││Creation││  Viz     ││  ing   │
└───┬────┘└───┬────┘└────┬─────┘└───┬────┘└────┬─────┘└───┬────┘
    │         │          │          │          │          │
    ▼         ▼          ▼          ▼          ▼          ▼
┌────────┐┌────────┐┌──────────┐┌────────┐┌──────────┐┌────────┐
│paper/  ││configs/││inference/││biomark-││results/  ││8640-sl6│
│sections││.cursor ││nuts.py   ││ers/    ││plots/    ││-bin/   │
│refs.bib││git,uv  ││prob_mod. ││pipeline││figures/  ││loaders │
│figures ││hydra   ││simulate  ││features││analysis  ││masks   │
└────────┘└────────┘└──────────┘└────────┘└──────────┘└────────┘
```

## Skill Interaction Patterns

### Data Flow (left to right)
```
Raw Data ──► Image Processing ──► Spectrum Estimation ──► Biomarker Creation
  .bin          load, mask,          NUTS sampler,          logistic regression,
  .json         subsample,           inverse problem,       LOOCV, AUC,
  .csv          signal extract       posterior samples      uncertainty propagation
                     │                      │                       │
                     └──────────────────────┴───────────────────────┘
                                            │
                                            ▼
                                   Benchmarking & Viz
                                   figures, tables, diagnostics
                                            │
                                            ▼
                                   MRM Paper Writing
                                   LaTeX sections, compilation
```

### Feedback Loop (iterative refinement)
```
Human feedback ──► Orchestrator ──► [activate relevant skill]
                                         │
                                         ▼
                                    Execute task
                                         │
                                         ▼
                                    Show results to human
                                         │
                                         ▼
                                    Human feedback ──► ...
```

## Skill Activation Triggers

| User says / Task involves | Skill activated |
|---------------------------|-----------------|
| "write the methods section" | mrm-paper-writing |
| "fix the git conflict" | general-coding |
| "run NUTS on this data" | spectrum-estimation |
| "compute AUC for PZ" | biomarker-creation |
| "create the ROC curve figure" | benchmarking-visualization |
| "load the binary images" | image-processing |
| "pixel-wise heatmap" | image-processing + spectrum-estimation + benchmarking-visualization |
| "convergence diagnostics figure" | spectrum-estimation + benchmarking-visualization |
| "direction independence analysis" | image-processing + spectrum-estimation |

## Cross-Skill Workflows

### Workflow A: Paper Section Drafting
1. **mrm-paper-writing**: Read outline, plan section structure
2. **spectrum-estimation** or **biomarker-creation**: Verify claims against code/results
3. **mrm-paper-writing**: Draft LaTeX content
4. Human: Review and feedback
5. **mrm-paper-writing**: Revise

### Workflow B: New Figure Generation
1. **benchmarking-visualization**: Design figure, identify data needs
2. **spectrum-estimation** or **biomarker-creation**: Run analysis if needed
3. **image-processing**: Prepare data if imaging data involved
4. **benchmarking-visualization**: Generate publication-quality figure
5. **mrm-paper-writing**: Add figure environment to LaTeX

### Workflow C: Pixel-wise Mapping (complex, multi-skill)
1. **image-processing**: Load .bin files, subsample to 64×64, create mask
2. **spectrum-estimation**: Benchmark single-pixel NUTS speed
3. **spectrum-estimation**: Run pixel-wise inference (batch)
4. **biomarker-creation**: Apply trained classifier per pixel
5. **benchmarking-visualization**: Generate heatmap overlaid on anatomy
6. **mrm-paper-writing**: Write results paragraph + figure

### Workflow D: Robustness Analysis
1. **spectrum-estimation**: Define synthetic spectra (including inverse)
2. **spectrum-estimation**: Run at multiple SNR levels
3. **benchmarking-visualization**: Create convergence diagnostics figure
4. **mrm-paper-writing**: Write methods/results paragraphs

## Blocked Items (requiring human action)
- [ ] Contact Stephan: all-directions data from Dropbox
- [ ] Contact Stephan: patient demographics table
- [ ] Contact Stephan: additional GGG case
- [ ] Contact Stephan: confirm b-value mapping for 46 images
- [ ] Install LaTeX: `brew install --cask basictex`
