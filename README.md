# DA_GestureRecognition-clean

## Project overview
This project bridges vision-based skeleton data (NTU RGB+D) and inertial motion capture data (Xsens). Both data streams are converted into **Local Relative Quaternions**; a Transformer encoder (`KinematicEncoder`) is pretrained on NTU under five objectives (scratch, mae, supMAE, supLP120/supervised, supcon), then transferred to a small, high-fidelity Xsens gesture dataset under subject-held-out few-shot calibration. External validity is checked on two public datasets (CZU-MHAD, UTD-MHAD).

## Status
This is a clean, from-scratch reproducibility rerun of [`RTHMLab/DA_GestureRecognition`](https://github.com/RTHMLab/DA_GestureRecognition) (branch `swing-mode-xsens`), forked **2026-07-13** to produce the final numbers for the paper with a verified, reproducible pipeline and no accumulated experimental debris. See `migration/MANIFEST.md` for exactly what was carried over from the original repo and why.

## Layout
- `scripts/` — the executable pipeline, organized by stage: `pretrain/` (5 NTU objectives), `data_pipeline/` (parsers producing `Data_Processed/`), `main_experiment/` (main Xsens LOSO + analysis), `external/{czu,utd}/` (external-dataset validation), `controller/` (downstream task-reliability simulation), `orchestration/` (`.sh` chains that run the above in order).
- `src/` — shared library: `models/` (`KinematicEncoder`/`KinematicDecoder`), `data/` (NTU loader/parser).
- `paper/` — the manuscript (`paper_results.md` = source of truth for numbers), IEEE LaTeX output.
- `wiki/` — project knowledge base; start at `wiki/index.md`.
- `tasks.md` — the work plan this rerun follows.
- `RESEARCH_LOG.md` — the Planning ↔ Implementation numbers channel.
- `migration/` — provenance record of what data was copied from the original repo (checksums for irreplaceable items, frozen dependency versions, environment info).

## Setup
```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
`requirements.txt` is a frozen snapshot from the original repo's environment (Python 3.12.3) — install exactly this before running anything, not latest versions, to keep results comparable.
