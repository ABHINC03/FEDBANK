"""
run.py — Pipeline orchestrator.
FEDBANK_preprocessing_spec.md Section 10.

Usage:
  python -m scripts.preprocess.run --stage all --sample 20000   # smoke test
  python -m scripts.preprocess.run --stage all                   # full run
  python -m scripts.preprocess.run --stage S3                    # single stage (from interim state)
  python -m scripts.preprocess.run --stage all --force           # force re-run of S0
"""
from __future__ import annotations
import os
os.environ.setdefault("CUDF_SPILL", "on")

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.preprocess.config import Config
from scripts.preprocess.gpu_env import init_gpu, free_gpu, vram
from scripts.preprocess.io_utils import csv_to_parquet, save_state, load_state


def parse_args():
    p = argparse.ArgumentParser(description="FEDBANK preprocessing pipeline")
    p.add_argument("--stage", default="all",
                   help="Stage to run: all | S0 | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | S10")
    p.add_argument("--sample", type=int, default=0,
                   help="Smoke-test sample size (0 = full run)")
    p.add_argument("--force", action="store_true",
                   help="Force re-run of S0 (CSV→Parquet conversion)")
    return p.parse_args()


def main():
    args = parse_args()

    cfg = Config()
    cfg.SAMPLE_N = args.sample

    stage = args.stage.upper()
    run_all = (stage == "ALL")

    print("=" * 70)
    print(f"FEDBANK Preprocessing Pipeline  |  stage={stage}  |  sample={args.sample}")
    print("=" * 70)

    t_pipeline = time.time()

    # ── S0: CSV → Parquet ────────────────────────────────────────────────
    if run_all or stage == "S0":
        try:
            csv_to_parquet(cfg, force=args.force)
        except Exception:
            print("[PIPELINE] S0 FAILED:")
            traceback.print_exc()
            sys.exit(1)

    if stage == "S0":
        print("[PIPELINE] Stage S0 complete.")
        return

    # ── Load GPU + import stages ─────────────────────────────────────────
    init_gpu()
    vram(tag=" pipeline-start")

    import cudf
    from scripts.preprocess import (
        s1_split, s2_clean, s3_engineer, s4_encode,
        s5_missing_flags, s6_winsor_skew, s7_impute,
        s8_scale, s9_reduce, s10_export, validate
    )

    # State management
    state_path = cfg.ARTIFACTS_DIR / "preprocess_state.json"

    # For single-stage runs, try to load existing state
    state: dict = {}
    if not run_all and state_path.exists():
        print(f"[PIPELINE] Loading existing state from {state_path}")
        state = load_state(state_path)

    stage_timings: dict[str, float] = {}

    def run_stage(name: str, fn, *args, **kwargs):
        if not run_all and stage != name:
            return None
        print(f"\n{'─'*60}")
        print(f"[PIPELINE] Running {name} …")
        t0 = time.time()
        try:
            result = fn(*args, **kwargs)
            elapsed = time.time() - t0
            stage_timings[name] = elapsed
            print(f"[PIPELINE] {name} completed in {elapsed:.1f}s")
            return result
        except Exception:
            print(f"[PIPELINE] {name} FAILED:")
            traceback.print_exc()
            sys.exit(1)

    # ── S1 ───────────────────────────────────────────────────────────────
    df = None
    if run_all or stage == "S1":
        t0 = time.time()
        print(f"\n{'─'*60}\n[PIPELINE] Running S1 …")
        try:
            df, state = s1_split.run(cfg, state)
            stage_timings["S1"] = time.time() - t0
        except Exception:
            traceback.print_exc(); sys.exit(1)
        # Save intermediate
        interim_path = cfg.INTERIM_DIR / "after_s1.parquet"
        df.to_parquet(interim_path)
        save_state(state, state_path)
        if stage == "S1":
            print("[PIPELINE] Stage S1 complete."); return

    # Resume from parquet if single stage
    if df is None:
        interim_path = cfg.INTERIM_DIR / "after_s1.parquet"
        if interim_path.exists():
            print(f"[PIPELINE] Loading interim from {interim_path}")
            df = cudf.read_parquet(interim_path)
        else:
            print("[PIPELINE] No interim state; re-running S1 first …")
            df, state = s1_split.run(cfg, state)
            df.to_parquet(cfg.INTERIM_DIR / "after_s1.parquet")
            save_state(state, state_path)

    def _checkpoint(df, tag, state):
        p = cfg.INTERIM_DIR / f"after_{tag}.parquet"
        df.to_parquet(p)
        save_state(state, state_path)
        print(f"[PIPELINE] Checkpointed → {p}")

    # ── S2 ───────────────────────────────────────────────────────────────
    if run_all or stage == "S2":
        t0 = time.time()
        print(f"\n{'─'*60}\n[PIPELINE] Running S2 …")
        try:
            df, state = s2_clean.run(cfg, df, state)
            stage_timings["S2"] = time.time() - t0
            _checkpoint(df, "s2", state)
        except Exception:
            traceback.print_exc(); sys.exit(1)
        if stage == "S2": print("[PIPELINE] Stage S2 complete."); return
    elif df is not None:
        p = cfg.INTERIM_DIR / "after_s2.parquet"
        if p.exists(): df = cudf.read_parquet(p)

    # ── S3 ───────────────────────────────────────────────────────────────
    if run_all or stage == "S3":
        t0 = time.time()
        print(f"\n{'─'*60}\n[PIPELINE] Running S3 …")
        try:
            df, state = s3_engineer.run(cfg, df, state)
            stage_timings["S3"] = time.time() - t0
            _checkpoint(df, "s3", state)
        except Exception:
            traceback.print_exc(); sys.exit(1)
        if stage == "S3": print("[PIPELINE] Stage S3 complete."); return
    elif df is not None:
        p = cfg.INTERIM_DIR / "after_s3.parquet"
        if p.exists(): df = cudf.read_parquet(p)

    # ── S4 ───────────────────────────────────────────────────────────────
    if run_all or stage == "S4":
        t0 = time.time()
        print(f"\n{'─'*60}\n[PIPELINE] Running S4 …")
        try:
            df, state = s4_encode.run(cfg, df, state)
            stage_timings["S4"] = time.time() - t0
            _checkpoint(df, "s4", state)
        except Exception:
            traceback.print_exc(); sys.exit(1)
        if stage == "S4": print("[PIPELINE] Stage S4 complete."); return
    elif df is not None:
        p = cfg.INTERIM_DIR / "after_s4.parquet"
        if p.exists(): df = cudf.read_parquet(p)

    # ── S5 ───────────────────────────────────────────────────────────────
    if run_all or stage == "S5":
        t0 = time.time()
        print(f"\n{'─'*60}\n[PIPELINE] Running S5 …")
        try:
            df, state = s5_missing_flags.run(cfg, df, state)
            stage_timings["S5"] = time.time() - t0
            _checkpoint(df, "s5", state)
        except Exception:
            traceback.print_exc(); sys.exit(1)
        if stage == "S5": print("[PIPELINE] Stage S5 complete."); return
    elif df is not None:
        p = cfg.INTERIM_DIR / "after_s5.parquet"
        if p.exists(): df = cudf.read_parquet(p)

    # ── S6 ───────────────────────────────────────────────────────────────
    if run_all or stage == "S6":
        t0 = time.time()
        print(f"\n{'─'*60}\n[PIPELINE] Running S6 …")
        try:
            df, state = s6_winsor_skew.run(cfg, df, state)
            stage_timings["S6"] = time.time() - t0
            _checkpoint(df, "s6", state)
        except Exception:
            traceback.print_exc(); sys.exit(1)
        if stage == "S6": print("[PIPELINE] Stage S6 complete."); return
    elif df is not None:
        p = cfg.INTERIM_DIR / "after_s6.parquet"
        if p.exists(): df = cudf.read_parquet(p)

    # ── S7 ───────────────────────────────────────────────────────────────
    if run_all or stage == "S7":
        t0 = time.time()
        print(f"\n{'─'*60}\n[PIPELINE] Running S7 …")
        try:
            df, state = s7_impute.run(cfg, df, state)
            stage_timings["S7"] = time.time() - t0
            _checkpoint(df, "s7", state)
        except Exception:
            traceback.print_exc(); sys.exit(1)
        if stage == "S7": print("[PIPELINE] Stage S7 complete."); return
    elif df is not None:
        p = cfg.INTERIM_DIR / "after_s7.parquet"
        if p.exists(): df = cudf.read_parquet(p)

    # ── S8 ───────────────────────────────────────────────────────────────
    if run_all or stage == "S8":
        t0 = time.time()
        print(f"\n{'─'*60}\n[PIPELINE] Running S8 …")
        try:
            df, state = s8_scale.run(cfg, df, state)
            stage_timings["S8"] = time.time() - t0
            _checkpoint(df, "s8", state)
        except Exception:
            traceback.print_exc(); sys.exit(1)
        if stage == "S8": print("[PIPELINE] Stage S8 complete."); return
    elif df is not None:
        p = cfg.INTERIM_DIR / "after_s8.parquet"
        if p.exists(): df = cudf.read_parquet(p)

    # ── S9 ───────────────────────────────────────────────────────────────
    if run_all or stage == "S9":
        t0 = time.time()
        print(f"\n{'─'*60}\n[PIPELINE] Running S9 …")
        try:
            df, state, tracks = s9_reduce.run(cfg, df, state)
            stage_timings["S9"] = time.time() - t0
            _checkpoint(df, "s9", state)
        except Exception:
            traceback.print_exc(); sys.exit(1)
        if stage == "S9": print("[PIPELINE] Stage S9 complete."); return
    else:
        p = cfg.INTERIM_DIR / "after_s9.parquet"
        if p.exists():
            df = cudf.read_parquet(p)
        tracks_path = cfg.ARTIFACTS_DIR / "preprocess_state.json"
        if tracks_path.exists():
            s = load_state(tracks_path)
            s9 = s.get("s9", {})
            tracks = {"track_a": s9.get("track_a",[]), "track_b": s9.get("track_b",[])}
        else:
            tracks = {"track_a": [], "track_b": []}

    # ── S10 ──────────────────────────────────────────────────────────────
    if run_all or stage == "S10":
        t0 = time.time()
        print(f"\n{'─'*60}\n[PIPELINE] Running S10 …")
        try:
            result = s10_export.run(cfg, df, state, tracks)
            stage_timings["S10"] = time.time() - t0
        except Exception:
            traceback.print_exc(); sys.exit(1)

    # ── Report ───────────────────────────────────────────────────────────
    if run_all:
        _write_report(cfg, state, stage_timings, tracks)
        print("\n[PIPELINE] Running validations …")
        try:
            validate.run_assertions(cfg)
        except Exception:
            traceback.print_exc()
            print("[PIPELINE] VALIDATION FAILED — see traceback above")

    total = time.time() - t_pipeline
    print(f"\n{'='*70}")
    print(f"[PIPELINE] Total time: {total:.1f}s")
    print(f"Stage timings: { {k: f'{v:.1f}s' for k,v in stage_timings.items()} }")
    free_gpu()
    vram(tag=" pipeline-end")


def _write_report(cfg: Config, state: dict, timings: dict, tracks: dict):
    import datetime
    s1 = state.get("s1", {})
    s2 = state.get("s2", {})
    s5 = state.get("s5", {})
    s7 = state.get("s7", {})
    s9 = state.get("s9", {})

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        f"# FEDBANK Preprocessing Report",
        f"",
        f"Generated: {now}  |  Sample: {cfg.SAMPLE_N if cfg.SAMPLE_N else 'full'}",
        f"",
        f"## Library Versions",
        f"",
    ]

    try:
        import cudf, cuml, cupy as cp, rmm
        lines += [
            f"| Library | Version |",
            f"|---|---|",
            f"| cuDF | {cudf.__version__} |",
            f"| cuML | {cuml.__version__} |",
            f"| cuPy | {cp.__version__} |",
            f"| RMM  | {rmm.__version__} |",
        ]
    except Exception:
        lines.append("(version info unavailable)")

    lines += [
        f"",
        f"## Split Table",
        f"",
        f"| Split | Rows | Expected | Fraud Rate | Paper Ref |",
        f"|---|---|---|---|---|",
    ]
    split_names  = {0:"train",1:"val",2:"test",3:"monitor"}
    paper_fraud  = {0:0.0333,1:0.0362,2:0.0370,3:0.0364}
    paper_rows   = {0:273000,1:58500,2:58500,3:200540}
    sizes = s1.get("split_sizes", {})
    for sv, sn in split_names.items():
        n = sizes.get(sv, sizes.get(str(sv), "?"))
        pf = paper_fraud[sv]
        pr = paper_rows[sv]
        lines.append(f"| {sn} | {n} | {pr} | (see pipeline output) | {pf:.2%} |")

    lines += [
        f"",
        f"## Stage Runtimes",
        f"",
        f"| Stage | Time (s) |",
        f"|---|---|",
    ] + [f"| {k} | {v:.1f} |" for k, v in timings.items()]

    lines += [
        f"",
        f"## Dropped Columns",
        f"",
        f"| Reason | Count |",
        f"|---|---|",
        f"| Missing rate > {cfg.DROP_MISSING} | {len(s2.get('drop_missing',[]))} |",
        f"| Constant (nunique ≤ 1) | {len(s2.get('drop_constant',[]))} |",
        f"| Near-constant (top share ≥ {cfg.NEAR_CONST_SHARE}) | {len(s2.get('drop_nearconst',[]))} |",
        f"| Hard prune (S9) | {len(s9.get('drop_hard',[]))} |",
        f"| Correlation prune (S9) | {len(s9.get('drop_corr',[]))} |",
    ]

    lines += [
        f"",
        f"## Imputation Counts",
        f"",
        f"| Method | Columns |",
        f"|---|---|",
        f"| No-op (0% missing) | (remainder) |",
        f"| KNN (k={cfg.KNN_K}) | {len(s7.get('knn_cols',[]))} |",
        f"| Median | {len(s7.get('median_cols',[]))} |",
        f"",
        f"KNN anchor columns ({len(s7.get('anchor_cols',[]))}):",
        f"`{', '.join(s7.get('anchor_cols',[]))}`",
        f"",
        f"## Missing Flags",
        f"",
        f"Original flaggable columns: {s5.get('flag_cols_before_dedup','?')}",
        f"Unique flags kept: {len(s5.get('added_flags',[]))}",
    ]

    track_a = tracks.get("track_a", [])
    track_b = tracks.get("track_b", [])
    importance = s9.get("importance", {})
    psi_vals   = s9.get("psi_vals", {})

    lines += [
        f"",
        f"## Track A — {len(track_a)} Features (sel)",
        f"",
        f"| Rank | Feature | Importance Gain | PSI(train→val) |",
        f"|---|---|---|---|",
    ]
    for i, col in enumerate(track_a, 1):
        imp = importance.get(col, 0.0)
        pv  = psi_vals.get(col, 0.0)
        lines.append(f"| {i} | {col} | {imp:.4f} | {pv:.4f} |")

    lines += [
        f"",
        f"## Track B — {len(track_b)} Features (pca)",
        f"",
        f"| # | Feature |",
        f"|---|---|",
    ] + [f"| {i} | {col} |" for i, col in enumerate(track_b, 1)]

    top10 = s9.get("top10_psi", [])
    lines += [
        f"",
        f"## Top-10 PSI Features (train→val)",
        f"",
        f"| Feature | PSI |",
        f"|---|---|",
    ] + [f"| {col} | {v:.4f} |" for col, v in top10]

    lines += [
        f"",
        f"## Deviations from the Paper",
        f"",
        f"| Topic | Paper | This implementation | Why |",
        f"|---|---|---|---|",
        f"| Split | normalized-DT boundaries | row-count boundaries matching paper Table 1 | exact boundaries not reproducible without their scaler |",
        f"| KNN imputation | sklearn KNNImputer | cuML brute-force KNN on {cfg.KNN_ANCHOR_MAX} anchor cols + {cfg.KNN_REF} train reference | no GPU KNNImputer; memory |",
        f"| Order | impute → transform | winsorize + transform → impute | distance quality under heavy tails |",
        f"| Skew transform | Yeo-Johnson | signed-log1p (YJ optional) | determinism / API risk |",
        f"| card_addr_hash | raw hash feature | frequency-encoded | raw hash magnitude is meaningless |",
        f"| Velocity / ratio | undefined | causal definitions in S3 | avoid future leakage |",
        f"| Time proxies | month/year listed | computed, excluded from features | cannot generalize under temporal split |",
        f"| 38 features | selected 'by analysis' | data-driven K={cfg.K_SEL} + PCA track | reproducible, reduces noise |",
        f"| SMOTE / graph | in paper | out of scope | modeling stage |",
        f"",
        f"---",
        f"*Report generated automatically by `scripts/preprocess/run.py`.*",
    ]

    report_path = cfg.REPORTS_DIR / "preprocess_report.md"
    cfg.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[REPORT] Written → {report_path}")


if __name__ == "__main__":
    main()
