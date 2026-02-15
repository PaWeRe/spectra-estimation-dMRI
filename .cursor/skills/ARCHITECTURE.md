# Agentic System Architecture

## Overview

A human-AI collaborative system for producing an MRM journal manuscript.
The human provides high-level direction; the agent handles all implementation.
The system uses **Skills** (domain knowledge) and **Subagents** (parallel workers).

## System Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        HUMAN (Patrick)                              │
│  Role: Strategic direction, scientific judgment, feedback           │
│  Interface: Terminal (Cursor chat)                                  │
│  Actions: Approve plans, give feedback, contact co-authors          │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                    high-level direction / feedback
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                 ORCHESTRATOR AGENT (Cursor/Claude)                   │
│                                                                     │
│  Role: Task decomposition, delegation, progress tracking            │
│  Context: .cursorrules + PAPER_PLAN.md + TODO list                  │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ SKILLS (.cursor/skills/) — Domain Knowledge                 │    │
│  │                                                             │    │
│  │ mrm-paper-writing   │ general-coding   │ spectrum-estimation│    │
│  │ biomarker-creation   │ bench-viz        │ image-processing  │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  Can delegate to SUBAGENTS for parallel/isolated work:              │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ SUBAGENTS (.cursor/agents/) — Parallel Workers               │   │
│  │                                                              │   │
│  │ ┌──────────┐ ┌─────────────┐ ┌──────────┐ ┌─────────────┐  │   │
│  │ │researcher│ │ computation │ │ verifier │ │latex-writer  │  │   │
│  │ │ (fast)   │ │ (background)│ │ (fast)   │ │ (inherit)    │  │   │
│  │ │ gather & │ │ run scripts │ │ validate │ │ draft LaTeX  │  │   │
│  │ │ verify   │ │ & analysis  │ │ & check  │ │ sections     │  │   │
│  │ └──────────┘ └─────────────┘ └──────────┘ └─────────────┘  │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                               │
                ┌──────────────┼──────────────┐
                ▼              ▼              ▼
         ┌──────────┐  ┌──────────┐  ┌──────────┐
         │  paper/  │  │   src/   │  │ results/ │
         │ LaTeX    │  │  Python  │  │ data &   │
         │ sections │  │  package │  │ figures  │
         └──────────┘  └──────────┘  └──────────┘
```

## Skills vs Subagents

| Aspect | Skills | Subagents |
|--------|--------|-----------|
| Purpose | Domain knowledge | Parallel execution |
| Location | `.cursor/skills/*/SKILL.md` | `.cursor/agents/*.md` |
| Activation | Loaded when relevant to task | Delegated by orchestrator |
| Context | Shared with main agent | Independent context window |
| Execution | Advisory (guide the agent) | Autonomous (do the work) |

### When to use what
- **"Write the theory section"** → Orchestrator reads `mrm-paper-writing` skill, delegates to `latex-writer` subagent
- **"Run the full pipeline"** → Orchestrator reads `spectrum-estimation` skill, delegates to `computation` subagent (background)
- **"Check if AUC numbers in the paper match results"** → Delegates to `researcher` subagent
- **"Verify the methods section is accurate"** → Delegates to `verifier` subagent
- **"Write intro AND run diagnostics"** → `latex-writer` and `computation` run in parallel

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
