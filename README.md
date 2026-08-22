# Adaptive Residual-Correction FEM

Reproducibility package for numerical experiments on adaptive finite elements, residual correction, active-subspace energy recovery, halo-localized residual correction (HLRC), and residual-controlled halo correction (RCHC) for scalar elliptic problems.

## Contents

- `code/solver.py` — production FEM/AFEM/HLRC/RCHC solver and benchmark definitions.
- `code/generate_results.py` — regenerates the primary numerical result tables.
- `code/verify_and_extend.py` — runs independent numerical checks and extended sensitivity studies.
- `code/validate_results.py` — checks archived CSV consistency and recovery identities.
- `code/make_figures.py` — regenerates the supplied figures from the archived CSV files.
- `code/timing_benchmark.py` — repeated final-level timing benchmark.
- `data/` — machine-readable numerical results.
- `figures/` — regenerated vector PDF and high-resolution PNG figures.
- `requirements.txt` — pinned Python dependencies.
- `environment.json` — numerical reproduction environment metadata.
- `timing_environment.json` — hardware/software metadata for the repeated timing audit.
- `validation/` — generated validation output.

## Installation

Python 3.13 was used for the archived runs. Install the pinned dependencies with:

```bash
python -m pip install -r requirements.txt
```

## Reproduce the numerical results

Run the complete workflow from the repository root:

```bash
bash run_reproduce.sh
```

Or run the components separately:

```bash
python code/generate_results.py --outdir regenerated_primary
python code/verify_and_extend.py --primary-dir regenerated_primary --outdir regenerated_extended
python code/validate_results.py
python code/make_figures.py
```

The first command regenerates the primary FEM/AFEM/local-correction tables. The second produces convergence slopes, Dörfler-marking sensitivity, the interface-estimator ablation, RCHC threshold sensitivity, and the numerical verification matrix. The validator checks the archived machine-readable results and exact algebraic/recovery identities. Figure generation uses the CSV archive in `data/`.

## Benchmarks

The package contains four benchmark classes:

1. smooth Poisson control;
2. L-shaped singularity;
3. high-contrast diffusion interface with coefficient jump `1e4`;
4. localized reaction-diffusion layer.

The adaptive workflow uses conforming linear triangular finite elements, residual-based error indicators, Dörfler marking, conformity-preserving refinement, sparse direct solves, and coefficient-aware residual weighting for the interface case.

## Verification

`data/verification_checks.csv` contains the archived numerical/software checks, including same-space correction/direct equivalence, restricted assembly, the active-subspace Pythagorean identity, matrix-free residual consistency, a linear P1 patch test, mesh-quality checks, and an independent finite-difference comparison.

Run:

```bash
python code/validate_results.py
```

to regenerate `validation/independent_validation.csv`.

## Timing data

`data/timing_final_25reps.csv` contains 25 measured repetitions after warm-up for final-level direct and localized solves. `timing_environment.json` records the associated computational environment. Timing values are implementation- and machine-specific diagnostics.

## Citation

Citation metadata are provided in `CITATION.cff`.

## License

See `LICENSE` in the repository root.
