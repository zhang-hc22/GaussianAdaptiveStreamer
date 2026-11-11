#!/usr/bin/env python3
"""Score one finished run, validate all sampled rows, then release its YUV."""
import os
from pathlib import Path
import sys
import time

from quality_store import QualityError, complete_and_remove_yuv, ensure_space, quality_ready, run_path


def quality_jobs(cfg):
    return max(1, min(os.cpu_count() or 1, cfg["server"].get("quality_jobs", 4)))


def quality_environment(cfg):
    env = dict(os.environ)
    jobs = str(quality_jobs(cfg))
    for key in ("TIGAS_QUALITY_JOBS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = jobs
    return env


def finish_quality(cfg, out, run, call, log):
    """call(cmd) returns (rc, captured output); caller owns subprocess cleanup."""
    server = cfg["server"]
    rel = run["rel"]
    out = str(Path(out).resolve())
    directory = run_path(out, rel)
    stride = server["metric_stride"]
    paper = server.get("paper_metrics", True) and run["mode"] != "webrtc"
    if not quality_ready(out, rel, stride, paper):
        # Fail without rerunning the measured session if its payload is missing.
        if run["mode"] == "webrtc" and not any((directory / "frames").glob("*.yuv")):
            raise QualityError(f"{rel}: incomplete quality CSV and no YUV; retained all metadata")
        ensure_space(out, server.get("min_free_gib", 5.0))
        steps = [("render_gt", ["exp/render_gt.py", out, "--stride", str(stride), "--resolution", server["resolution"], "--run-rel", rel]),
                 ("metrics", ["exp/metrics.py", out, "--stride", str(stride), "--run-rel", rel,
                              "--jobs", str(quality_jobs(cfg))])]
        if paper:
            steps.append(("metrics_paper", ["exp/metrics_paper.py", out, "--stride", str(stride), "--run-rel", rel]))
        for step, args in steps:
            started = time.monotonic()
            rc, text = call([sys.executable, *args])
            with open(os.path.join(out, step + ".log"), "a") as fh:
                fh.write(f"\n--- {rel} ---\n{text}")
            log(f"quality {rel}: {step} {'done' if rc == 0 else 'FAILED'} ({time.monotonic() - started:.0f} s)")
            if rc:
                raise QualityError(f"{rel}: {step} failed; YUV retained; see {out}/{step}.log")
    result = complete_and_remove_yuv(out, rel, stride, paper)
    log(f"quality {rel}: all sampled rows verified; removed {result['deleted_files']} YUV files "
        f"({result['deleted_bytes'] / 2**20:.1f} MiB); data and statistics retained")
    return result
