---
name: general-coding
description: Handle general software engineering tasks including git operations, dependency management, project configuration, CI/CD, and integration with external services (W&B, MCP servers). Use when the task involves git branching, committing, package management with uv, Hydra configuration, or W&B experiment tracking.
compatibility: Requires uv package manager, git. Optional W&B CLI for experiment tracking.
---

# General Coding

## When to use this skill
Use for any software engineering task that isn't specific to a particular domain skill:
- Git operations (branching, committing, merging)
- Dependency management with `uv`
- Hydra configuration management
- W&B experiment tracking integration
- Project structure and organization
- Debugging build/runtime issues

## Package management
ALWAYS use `uv`:
```bash
uv run python script.py      # Run scripts
uv add package-name           # Add dependencies
uv sync                       # Install from lock file
```

## Project structure
```
src/spectra_estimation_dmri/  # Main package
├── data/                     # Data loaders and models
├── models/                   # Probabilistic model definition
├── inference/                # NUTS, Gibbs, MAP samplers
├── biomarkers/               # Classification pipeline
├── visualization/            # Plotting modules
├── analysis/                 # Sampler comparison
├── simulation/               # Synthetic data generation
└── utils/                    # Helpers

configs/                      # Hydra YAML configs
scripts/                      # Standalone analysis scripts
results/                      # Output data and figures
paper/                        # LaTeX manuscript
```

## Hydra configuration
- Main config: `configs/config.yaml`
- Override via CLI: `uv run python -m spectra_estimation_dmri.main dataset=bwh inference=nuts`
- Key configs: `dataset/bwh.yaml`, `inference/nuts.yaml`, `prior/ridge.yaml`

## Git workflow
- Branch: `paper/mrm-manuscript` (current working branch)
- Main branch: `main` (keep clean, don't merge until paper is ready)
- Commit frequently with descriptive messages
- Never force-push to main

## W&B integration
- Project: `bayesian-dMRI-biomarker`
- Set `local: true` in config to skip W&B logging during development
- Set `local: false` for tracked experiments

## Key patterns
- All results should be reproducible from code + config
- Save intermediate results to `results/` directory
- Use `seed: 42` for reproducibility
- Use Hydra's `recompute: false` to skip re-running cached inference
