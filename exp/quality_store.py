#!/usr/bin/env python3
"""Validate and durably checkpoint per-run quality before removing YUV data."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile
import time

METRICS = ("psnr", "ssim", "vmaf", "vif", "lpips", "dists")


class QualityError(RuntimeError):
    pass


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sync_file(path):
    with open(path, "rb") as fh:
        os.fsync(fh.fileno())


def sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", newline="", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as fh:
        tmp = Path(fh.name)
        try:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            fh.flush()
            os.fsync(fh.fileno())
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    try:
        os.replace(tmp, path)
        sync_dir(path.parent)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as fh:
        tmp = Path(fh.name)
        try:
            json.dump(value, fh, indent=2)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    try:
        os.replace(tmp, path)
        sync_dir(path.parent)
    finally:
        tmp.unlink(missing_ok=True)


def run_path(results, rel):
    base = (Path(results) / "runs").resolve()
    relative = Path(rel)
    if relative.is_absolute() or ".." in relative.parts:
        raise QualityError(f"invalid run path: {rel}")
    path = base / relative
    if not path.resolve().is_relative_to(base):
        raise QualityError(f"run escapes results directory: {rel}")
    return path


def expected_samples(directory, stride, paper=False):
    """Coverage comes from the immutable viewpoint log, including stand-ins."""
    directory = Path(directory)
    if stride < 1:
        raise QualityError("quality stride must be positive")
    run = json.loads((directory / "run.json").read_text())
    if run.get("status") != "ok":
        raise QualityError(f"{directory}: run did not succeed")
    frames = [json.loads(line) for line in (directory / "frames.ndjson").read_text().splitlines()]
    ids = [int(frame["i"]) for frame in frames]
    if len(ids) != run["viewpoints"] or len(ids) != len(set(ids)) or any(i < 0 or i >= run["viewpoints"] for i in ids):
        raise QualityError(f"{directory}: invalid viewpoint log")
    webrtc = run["args"].get("mode") == "webrtc"
    own = {int(f["i"]): f for f in frames if "error" not in f}
    expected = {}
    for f in frames:
        i = int(f["i"])
        if i % stride:
            continue
        if paper:
            if "error" not in f:
                expected[i] = False
        elif webrtc:
            if f.get("dec") is not None:
                expected[i] = False
            elif "standin_i" in f:
                expected[i] = True
        elif "error" not in f:
            expected[i] = False
        elif f["error"] == "skipped" and f.get("standin_i") in own:
            expected[i] = True
    if not expected:
        raise QualityError(f"{directory}: no scoreable sampled viewpoints")
    return expected


def validate_csv(path, expected, *, require_standin=False, allow_extra=False):
    path = Path(path)
    try:
        with path.open(newline="") as fh:
            reader = csv.DictReader(fh)
            required = {"i", *METRICS} | ({"standin"} if require_standin else set())
            if not required.issubset(reader.fieldnames or []):
                raise QualityError(f"{path}: missing quality columns")
            seen = set()
            for row in reader:
                i = int(row["i"])
                if i in seen or (i not in expected and not allow_extra):
                    raise QualityError(f"{path}: duplicate or unexpected sample {i}")
                seen.add(i)
                if not all(math.isfinite(float(row[key])) for key in METRICS):
                    raise QualityError(f"{path}: nonfinite metric for sample {i}")
                if i in expected and "standin" in row and int(row["standin"]) != int(expected[i]):
                    raise QualityError(f"{path}: wrong stand-in flag for sample {i}")
        if (not allow_extra and seen != set(expected)) or not set(expected).issubset(seen):
            raise QualityError(f"{path}: incomplete samples ({len(seen)}/{len(expected)})")
        return len(seen)
    except (OSError, ValueError, TypeError, csv.Error) as exc:
        raise QualityError(f"{path}: invalid quality CSV: {exc}") from exc


def quality_tables(results, rel, stride, paper=False):
    directory = run_path(results, rel)
    run = json.loads((directory / "run.json").read_text())
    required = [(Path(results) / "metrics" / (rel + ".csv"), expected_samples(directory, stride),
                 run["args"].get("mode") == "webrtc" or run["args"].get("pacing") == "tms")]
    if paper and run["args"].get("mode") != "webrtc":
        required.append((Path(results) / "metrics_paper" / (rel + ".csv"), expected_samples(directory, stride, paper=True), False))
    proofs = {}
    for path, expected, standin in required:
        rows = validate_csv(path, expected, require_standin=standin)
        if path.stat().st_mtime_ns < (directory / "run.json").stat().st_mtime_ns:
            raise QualityError(f"{path}: quality CSV predates run.json")
        proofs[str(path.relative_to(results))] = dict(sha256=sha(path), rows=rows)
    return proofs


def quality_ready(results, rel, stride, paper=False):
    try:
        quality_tables(results, rel, stride, paper)
        return True
    except (QualityError, OSError, ValueError):
        return False


def complete_and_remove_yuv(results, rel, stride, paper=False):
    """The durable quality proof is written BEFORE the first YUV is removed."""
    results = Path(results).resolve()
    directory = run_path(results, rel)
    proofs = quality_tables(results, rel, stride, paper)
    fingerprint = dict(run_sha256=sha(directory / "run.json"), frames_sha256=sha(directory / "frames.ndjson"),
                       stride=stride, paper_metrics=paper, quality=proofs)
    marker = directory / "quality_complete.json"
    try:
        previous = json.loads(marker.read_text()) if marker.is_file() else {}
    except (OSError, ValueError) as exc:
        raise QualityError(f"{marker}: invalid checkpoint; retain all remaining YUV") from exc
    same = all(previous.get(k) == value for k, value in fingerprint.items())
    frames = directory / "frames"
    if frames.is_symlink():
        raise QualityError(f"{frames}: refusing to clean a symlinked frame directory")
    current = []
    for path in sorted(frames.glob("*.yuv")):
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or not re.fullmatch(r"\d+(?:_standin)?\.yuv", path.name):
            raise QualityError(f"{path}: unexpected YUV file; retain all frames")
        current.append(dict(name=path.name, bytes=info.st_size, inode=info.st_ino, device=info.st_dev, mtime_ns=info.st_mtime_ns))
    if same and previous.get("yuv_removed") and not current:
        return dict(cached=True, deleted_files=0, deleted_bytes=0)
    if previous and not same and not {x["name"] for x in previous.get("yuv_files", [])}.issubset({x["name"] for x in current}):
        raise QualityError(f"{directory}: quality provenance changed after YUV cleanup began")
    planned = previous.get("yuv_files", []) if same else current
    if same:
        known = {item["name"]: item for item in planned}
        if any(known.get(item["name"]) != item for item in current):
            raise QualityError(f"{directory}: YUV changed after quality was checkpointed")
    record = dict(version=1, state="quality_complete", **fingerprint, yuv_files=planned,
                  yuv_removed=False, updated_at=time.time())
    for path in proofs:
        sync_file(results / path)
    sync_file(directory / "run.json")
    sync_file(directory / "frames.ndjson")
    atomic_json(marker, record)
    deleted_bytes = 0
    for item in current:
        path = frames / item["name"]
        info = path.lstat()
        if (info.st_size, info.st_ino, info.st_dev, info.st_mtime_ns) != (item["bytes"], item["inode"], item["device"], item["mtime_ns"]):
            raise QualityError(f"{path}: changed before deletion")
        path.unlink()
        deleted_bytes += item["bytes"]
    if frames.exists():
        sync_dir(frames)
    record.update(yuv_removed=True, updated_at=time.time())
    atomic_json(marker, record)
    return dict(cached=not current, deleted_files=len(current), deleted_bytes=deleted_bytes)


def remove_warmup_yuv(results, directory):
    """Remove successful warm-up payloads, which never enter measured quality."""
    base = Path(results).resolve() / "warmup"
    directory = Path(os.path.abspath(directory))
    if not directory.is_relative_to(base) or directory == base:
        raise QualityError(f"{directory}: not a warm-up run")
    for path in (directory, *directory.parents):
        if path.is_symlink():
            raise QualityError(f"{path}: refusing a symlinked warm-up directory")
        if path == base:
            break
    meta = directory / "run.json"
    if json.loads(meta.read_text()).get("status") != "ok":
        raise QualityError(f"{directory}: warm-up did not succeed; retain YUV")
    frames = directory / "frames"
    if frames.is_symlink():
        raise QualityError(f"{frames}: refusing a symlinked frame directory")
    current = []
    for path in sorted(frames.glob("*.yuv")):
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or not re.fullmatch(r"\d+(?:_standin)?\.yuv", path.name):
            raise QualityError(f"{path}: unexpected warm-up YUV; retain all frames")
        current.append(dict(name=path.name, bytes=info.st_size, inode=info.st_ino,
                            device=info.st_dev, mtime_ns=info.st_mtime_ns))
    marker = directory / "warmup_cleanup.json"
    fingerprint = sha(meta)
    try:
        previous = json.loads(marker.read_text()) if marker.exists() else {}
    except (OSError, ValueError) as exc:
        raise QualityError(f"{marker}: invalid warm-up cleanup checkpoint") from exc
    if previous:
        known = {item["name"]: item for item in previous["yuv_files"]}
        if previous.get("run_sha256") != fingerprint or any(known.get(item["name"]) != item for item in current):
            raise QualityError(f"{directory}: warm-up changed after cleanup began")
        if previous.get("yuv_removed") and not current:
            return dict(cached=True, deleted_files=0, deleted_bytes=0)
    record = dict(version=1, state="warmup_complete", reason="not_in_measured_statistics",
                  run_sha256=fingerprint, yuv_files=previous.get("yuv_files", current),
                  yuv_removed=False, updated_at=time.time())
    sync_file(meta)
    atomic_json(marker, record)
    for item in current:
        path = frames / item["name"]
        info = path.lstat()
        if (info.st_size, info.st_ino, info.st_dev, info.st_mtime_ns) != (item["bytes"], item["inode"], item["device"], item["mtime_ns"]):
            raise QualityError(f"{path}: changed before deletion")
        path.unlink()
    if frames.exists():
        sync_dir(frames)
    record.update(yuv_removed=True, updated_at=time.time())
    atomic_json(marker, record)
    return dict(cached=not current, deleted_files=len(current), deleted_bytes=sum(item["bytes"] for item in current))


def ensure_space(results, min_free_gib=5.0, temp_root=None):
    if min_free_gib <= 0:
        raise ValueError("minimum free disk space must be positive")
    roots = [Path(results).resolve(), Path(temp_root or tempfile.gettempdir()).resolve()]
    required = int(min_free_gib * 2**30)
    checked = []
    for path in roots:
        fs = os.statvfs(path)
        available = max(0, fs.f_bavail * fs.f_frsize)
        if available < required:
            raise QualityError(f"disk space low at {path}: {available / 2**30:.2f} GiB available, "
                               f"need {min_free_gib:g} GiB; stopped before the next run/quality step")
        checked.append(dict(path=str(path), available_bytes=available, required_bytes=required))
    return checked
