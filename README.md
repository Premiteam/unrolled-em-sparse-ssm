# Unrolled EM with Learned Elementwise Extrapolation for Sparse State Space Identification

Code for the paper

> R. Li, D. Chen, R. Zhang and F. Chen, "Unrolled Expectation Maximization with Learned Elementwise Extrapolation for Sparse State Space Identification."

The repository contains the implementation of the proposed method (unrolled EM with a shared elementwise extrapolation network, safeguard and handover to a bounded SQUAREM), the classical baselines (EM, SQUAREM in several variants, DAAREM, quasi-Newton EM), every script used to generate the simulation results of the main text and the supplementary material, the script that computes every reported number from the raw results, and the TikZ/pgfplots sources of all figures.

## Contents

| Path | Content |
|---|---|
| [`unrolled_em/`](unrolled_em) | method, baselines and experiment scripts |
| [`paper/numbers.txt`](paper/numbers.txt), [`paper/numbers.json`](paper/numbers.json) | every table and every number of the paper, as computed by [`unrolled_em/experiments/paper.py`](unrolled_em/experiments/paper.py) from the raw results |
| [`paper/data/`](paper/data) | data of Figs. 2 to 4 (written by the same script) |
| [`paper/figures/`](paper/figures), [`paper/figures.tex`](paper/figures.tex), [`paper/figures.pdf`](paper/figures.pdf) | pgfplots/TikZ sources of Figs. 1 to 4 and the compiled figures |
| [`pretrained/`](pretrained) | trained layers of all experiments (PyTorch state dicts and training logs) |
| [`tests/`](tests) | unit tests |
| [`scripts/`](scripts) | smoke test, full reproduction, and summary from the raw results (Windows `.bat` and POSIX `.sh`) |

The raw per-system results (about 70 MB compressed) are archived separately as `unrolled-em-sparse-ssm-results.tar.gz` (SHA-256 `8a81bcf087a0670d2f36306a7262e31b216e974742768a714b2ec5b6036a7488`). Extracting it in the repository root creates `results/` exactly as the experiment scripts write it; `results/README.md` in the archive describes its structure and file formats.

## Requirements

- Python 3.11
- NumPy, SciPy, PyTorch (CPU is sufficient)
- for the figures: a TeX distribution with TikZ and pgfplots

The results in the paper were computed with Python 3.11.1, NumPy 2.4.2, SciPy 1.17.1 and PyTorch 2.11.0 on Windows. Install the dependencies with [`requirements.txt`](requirements.txt)

```bash
pip install -r requirements.txt
```

or with [`environment.yml`](environment.yml)

```bash
conda env create -f environment.yml
conda activate unrolled-em
```

All commands are run from the repository root. No installation of the package itself is needed.

## Three levels of reproduction

**1. Figures from the included data.** The data of every figure is in [`paper/data/`](paper/data), and [`paper/figures.tex`](paper/figures.tex) compiles all figures.

```bash
cd paper
latexmk -pdf figures.tex
```

**2. All numbers and figures from the raw results.** Extract the results archive in the repository root, then run [`scripts/summarize.sh`](scripts/summarize.sh) or [`scripts/summarize.bat`](scripts/summarize.bat).

```bash
bash scripts/summarize.sh          # Linux / macOS
scripts\summarize.bat              # Windows
```

This runs [`export.py`](unrolled_em/experiments/export.py) and [`paper.py`](unrolled_em/experiments/paper.py), which print every table of the main text and the supplementary material together with the numbers quoted in the text, write them to [`paper/numbers.txt`](paper/numbers.txt) and [`paper/numbers.json`](paper/numbers.json), and regenerate [`paper/data/`](paper/data). Fig. 2 is recomputed by rerunning the four methods on the selected test system (a few minutes); everything else is read from the stored results. Afterwards compile the figures as in level 1.

**3. Everything from scratch.** [`scripts/reproduce.sh`](scripts/reproduce.sh) and [`scripts/reproduce.bat`](scripts/reproduce.bat) run all experiments in the order below and finish with level 2. The full run takes several days of CPU time. The number of parallel processes is set by the environment variable `WORKERS` (default 10). Timing steps run in a single process and should be run on an otherwise idle machine. Every job writes one file per system, so an interrupted run is resumed by repeating the same command; finished jobs are skipped. Every random system, record and data split is generated from the seed root in [`unrolled_em/protocol.json`](unrolled_em/protocol.json), so the same systems are obtained on every run. To reuse the trained layers instead of training them again, copy the contents of [`pretrained/`](pretrained) into `results/` before the run; training jobs whose model file exists are skipped.

A reduced version of the main pipeline (small dimensions, few systems, 20 training steps) runs in a few minutes and checks the installation, together with the unit tests in [`tests/`](tests). See [`scripts/smoke.sh`](scripts/smoke.sh) and [`scripts/smoke.bat`](scripts/smoke.bat).

```bash
python -m pytest -q tests
bash scripts/smoke.sh              # or scripts\smoke.bat
```

## Numerical reproducibility

All randomness is seeded, so a rerun on the same platform reproduces the results exactly. The iterations compare objective values whose differences can be close to machine precision late in a run, for example in the decrease test of the safeguard and of the switch to exact E-steps. With a different BLAS, NumPy or SciPy build these comparisons can occasionally go the other way, which changes the trajectory of a single system from that point on. The stored results were computed with the versions listed above. When 139 of the 150 runs of the switch experiment were repeated on Linux with NumPy 2.5.3 and SciPy 1.18.1, the numbers of E-steps to both predictive targets were identical on 138 systems, and the E-step of the switch differed on 11. Individual numbers can therefore differ slightly from the paper on another platform, while [`scripts/summarize.sh`](scripts/summarize.sh) applied to the archived results reproduces them exactly.

## Repository layout

### [`unrolled_em/`](unrolled_em)

| File | Content |
|---|---|
| [`core.py`](unrolled_em/core.py) | linear Gaussian state space model, system generator, Kalman filter and RTS smoother, exact and steady state E-step, monotone FISTA M-step, likelihood, SQUAREM, initial estimates |
| [`anytime.py`](unrolled_em/anytime.py) | E-step counting and anytime evaluation of an iteration |
| [`daarem.py`](unrolled_em/daarem.py) | DAAREM (translated from the reference implementation) |
| [`accel.py`](unrolled_em/accel.py) | direct optimizers of the preliminary tests (OWL-QN, SpaRSA) and Anderson acceleration |
| [`unroll.py`](unrolled_em/unroll.py) | learned layer: features, shared perceptron, bounded step and momentum, training loss |
| [`handcrafted.py`](unrolled_em/handcrafted.py) | classical elementwise steps for the ablation (Aitken, secant, missing information, constants) |
| [`protocol.json`](unrolled_em/protocol.json) | fixed protocol constants (seed root, noise variance, penalty grid, reference solver) |

### [`unrolled_em/experiments/`](unrolled_em/experiments)

| File | Content |
|---|---|
| [`common.py`](unrolled_em/experiments/common.py) | system families, data generation, scale estimate, penalty, accuracy targets, statistics |
| [`method.py`](unrolled_em/experiments/method.py) | proposed method: features, safeguarded layer, unrolled run, layer training |
| [`run.py`](unrolled_em/experiments/run.py) | main pipeline: penalty selection, reference solutions, training, tuning, test sets, shifts, sizes, timing |
| [`export.py`](unrolled_em/experiments/export.py) | summary of the main pipeline (`results/v2/verdict.json`) |
| [`exp_hand.py`](unrolled_em/experiments/exp_hand.py) | safeguarded bounded SQUAREM used by the handover |
| [`exp_cont.py`](unrolled_em/experiments/exp_cont.py) | final algorithm (trained layers + handover) on all test sets |
| [`exp_sqbt.py`](unrolled_em/experiments/exp_sqbt.py) | SQUAREM variants (backtracking SqS1 to SqS3, translation of the reference implementation) |
| [`exp_sqpkg_rest.py`](unrolled_em/experiments/exp_sqpkg_rest.py) | reference implementation of SQUAREM in the remaining comparisons |
| [`exp_review.py`](unrolled_em/experiments/exp_review.py) | steady state switch, SQUAREM with gamma_0 = 16, quasi-Newton EM, stopping rule, n = 512 |
| [`exp_time16.py`](unrolled_em/experiments/exp_time16.py) | timing of SQUAREM started with gamma_0 = 16 |
| [`exp_class.py`](unrolled_em/experiments/exp_class.py) | class-wise constants and linear-feature variants of the ablation |
| [`exp_lam.py`](unrolled_em/experiments/exp_lam.py) | penalty selected for support recovery |
| [`exp_ab.py`](unrolled_em/experiments/exp_ab.py) | GraphEM benchmark and fully solved M-steps |
| [`exp_ab_summary.py`](unrolled_em/experiments/exp_ab_summary.py) | summary of `exp_ab.py` |
| [`group3.py`](unrolled_em/experiments/group3.py) | native n = 256 model, n = 512 transfer, heat conduction networks |
| [`sq_diag.py`](unrolled_em/experiments/sq_diag.py) | Jacobian spectrum of the EM map and SQUAREM steplength statistics |
| [`nf_sens.py`](unrolled_em/experiments/nf_sens.py) | number of FISTA iterations in the M-step |
| [`smax_exp.py`](unrolled_em/experiments/smax_exp.py) | step bound s_max = 20 |
| [`dare_check.py`](unrolled_em/experiments/dare_check.py) | detectability and Riccati check under partial observation |
| [`check_switch.py`](unrolled_em/experiments/check_switch.py) | decrease condition for classical updates with the steady state E-step |
| [`ana_timing2.py`](unrolled_em/experiments/ana_timing2.py) | summary of the single process timing |
| [`paper.py`](unrolled_em/experiments/paper.py) | every table, number and figure data of the paper from the raw results |

### [`unrolled_em/preliminary/`](unrolled_em/preliminary)

| File | Content |
|---|---|
| [`classical.py`](unrolled_em/preliminary/classical.py) | preliminary comparison of classical EM variants |
| [`direct.py`](unrolled_em/preliminary/direct.py) | preliminary comparison with direct optimizers |

## Experiments and where they appear in the paper

Section, table and figure numbers refer to the main text (Sections I to VI, Tables I to V, Figs. 1 to 3) and the supplementary material (Sections VII to X, Tables VI to XI, Fig. 4). Each step is a module run as `python -u -m unrolled_em.<module> ...`; [`scripts/reproduce.sh`](scripts/reproduce.sh) lists the exact arguments.

| Step | Module | Used for |
|---|---|---|
| 1 | [`experiments.run`](unrolled_em/experiments/run.py) (`--part all`, `timing`, `percall`), [`experiments.export`](unrolled_em/experiments/export.py) | penalty selection, reference solutions, training, DAAREM tuning, test sets I and II, distribution shift, n = 256, GraphEM benchmark, ideal steps; Table II, Sections II-B, IV-A, V |
| 2 | [`experiments.group3`](unrolled_em/experiments/group3.py) | native n = 256 model, n = 512, heat conduction networks (Sections IV-D, IV-E, X-G) |
| 3 | [`experiments.exp_ab`](unrolled_em/experiments/exp_ab.py), [`experiments.exp_ab_summary`](unrolled_em/experiments/exp_ab_summary.py) | GraphEM outer iteration and fully solved M-steps (Sections IV-B, IX, X-D) |
| 4 | [`experiments.exp_lam`](unrolled_em/experiments/exp_lam.py) | penalty selected for support recovery (Sections IV-D, X-F) |
| 5 | [`experiments.exp_class`](unrolled_em/experiments/exp_class.py) | class-wise and linear variants of the ablation (Sections IV-C, X-E) |
| 6 | [`experiments.exp_sqbt`](unrolled_em/experiments/exp_sqbt.py) | SQUAREM variants (Table VI, Section X-C) |
| 7 | [`experiments.exp_sqpkg_rest`](unrolled_em/experiments/exp_sqpkg_rest.py) | reference implementation of SQUAREM in all other comparisons |
| 8 | [`experiments.smax_exp`](unrolled_em/experiments/smax_exp.py) | step bound s_max = 20 (Section X-E) |
| 9 | [`experiments.exp_cont`](unrolled_em/experiments/exp_cont.py) | final algorithm with handover: choice of nu and gamma_0, Tables III to V, VII, X, XI, Figs. 2 and 3, transfer, n = 512, heat conduction, timing |
| 10 | [`experiments.exp_review`](unrolled_em/experiments/exp_review.py) | steady state switch (Section VIII), SQUAREM with gamma_0 = 16, quasi-Newton EM, stopping rule (Table VIII), n = 512 |
| 11 | [`experiments.exp_time16`](unrolled_em/experiments/exp_time16.py) | time ratios against both versions of SQUAREM (Sections IV-B, X-D) |
| 12 | [`experiments.sq_diag`](unrolled_em/experiments/sq_diag.py) | Jacobian spectrum (Table IX) and SQUAREM steplengths (Sections II-C, X-C) |
| 13 | [`experiments.nf_sens`](unrolled_em/experiments/nf_sens.py) | number of FISTA iterations (Section IX) |
| 14 | [`experiments.dare_check`](unrolled_em/experiments/dare_check.py) | detectability under partial observation (Section VIII) |
| 15 | [`experiments.check_switch`](unrolled_em/experiments/check_switch.py) | decrease condition with the steady state E-step (Section VIII) |
| 16 | [`experiments.ana_timing2`](unrolled_em/experiments/ana_timing2.py) | single process timing of all compared methods |
| 17 | [`preliminary.classical`](unrolled_em/preliminary/classical.py), [`preliminary.direct`](unrolled_em/preliminary/direct.py) | preliminary tests with direct optimizers (Sections II-C, IX) |
| 18 | [`experiments.paper`](unrolled_em/experiments/paper.py) | all tables and numbers, data of Figs. 2 to 4 |

Setting names follow the paper: `F32_500`, `F64_500` and `F128_500` are fully observed systems with n = 32, 64 and 128 and T = 500; `P32_500` and `P64_500` observe the first n/2 states. In the output files the proposed method is `learned_tied_L4`, EM is `EM_ss_K50`, DAAREM is `DAAREM_tun`, the reference implementation of SQUAREM is `SQpkg_ss` (`SQpkg16_ss` when started with gamma_0 = 16), and `SQ_ss_K50` is SQUAREM with immediate fallback.

## Citation

Citation metadata is in [`CITATION.cff`](CITATION.cff).

## License

This program is released under the GNU General Public License version 2 (`GPL-2.0-only`, see [`LICENSE`](LICENSE)). The files [`unrolled_em/daarem.py`](unrolled_em/daarem.py) and [`unrolled_em/experiments/exp_sqbt.py`](unrolled_em/experiments/exp_sqbt.py) contain code translated from the R packages daarem and SQUAREM, which are distributed under the GPL; see [`NOTICE`](NOTICE) for details.
