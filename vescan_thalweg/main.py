"""Thalweg-based pipeline from the haversack store (no VMTK): one or more patients, as
vescan_vmtk/main.py. Everything is set in the config (see configs/thalweg/example.json and
vescan_thalweg/config.py). Launched from the repo root:

    python main_thalweg.py --config configs/thalweg/example.json [--structures artery veins]

This module loops over the patients; for each one vescan_thalweg.orchestrator.run_patient runs the
stages enabled in pipeline.stages. Messages go to the console and to
<results>/<patient>/pipeline.log (logging section of the config).

--jobs N processes N patients at a time, one process each (the stages use one core). Each process
logs only to its patient's pipeline.log; the console shows one line per finished patient.
Threads: each process uses (usable cores) / N of them, the usable cores being the container's CPU
quota when there is one (a Jupyter container can see 128 cores and be allowed 16).
In both modes a progress bar line (done/total, elapsed, ETA) is shown at the start and after each
finished patient; with --jobs also every minute while the patients run.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import nullcontext
from pathlib import Path

from vescan_thalweg import config as cfgmod
from vescan_thalweg import log, orchestrator

logger = logging.getLogger("vescan_thalweg.main")


def iter_patients(batch):
    """(name, patient folder, surfaces folder, results folder) as VeScan: one patient if
    batch.enabled is false, otherwise every subfolder that contains the store."""
    root, surfaces, results = (Path(batch.patients_root_dir), Path(batch.surfaces_root_dir),
                               Path(batch.results_root_dir))
    if not batch.enabled:
        yield root.name, root, surfaces, results
        return
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if (d / batch.store).exists():
            yield d.name, d, surfaces / d.name, results / d.name


def run_one(cfg, patient, patient_dir, surfaces_dir, results_dir, structures) -> bool:
    """One patient, with its messages also in <results>/<patient>/pipeline.log. False if it failed
    in batch mode (outside batch mode the error is raised)."""
    results_dir.mkdir(parents=True, exist_ok=True)
    file_log = log.to_file(results_dir / cfg.logging.file_name) if cfg.logging.file_name else nullcontext()
    with file_log:
        logger.info(f"=== patient {patient} ({patient_dir}) -> {results_dir}")
        try:
            orchestrator.run_patient(cfg, patient, patient_dir, surfaces_dir, results_dir, structures)
            return True
        except Exception:
            if not cfg.batch.enabled:
                raise
            logger.exception(f"patient {patient} failed, moving on to the next one")
            return False


def usable_cpus() -> int:
    """The cores this process may use: the cgroup CPU quota if there is one, otherwise the cores
    it is allowed to run on."""
    try:
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()        # cgroup v2
        if quota != "max":
            return max(1, int(quota) // int(period))
    except (OSError, ValueError):
        pass
    try:
        quota = int(open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read())       # cgroup v1
        period = int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read())
        if quota > 0:
            return max(1, quota // period)
    except (OSError, ValueError):
        pass
    return len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1


def limit_threads(threads: int):
    """At most `threads` threads for the multithreaded steps: the kd-tree queries and zarr's
    decoding of the store."""
    import zarr
    from vescan_thalweg import threads as threads_mod
    threads_mod.KDTREE_WORKERS = threads
    zarr.config.set({"threading.max_workers": threads})


def _worker(cfg, patient_args, threads):
    """A worker process: its messages go only to the patient's pipeline.log (console if none)."""
    limit_threads(threads)
    root = logging.getLogger(log.ROOT)
    root.handlers.clear()                      # a forked worker inherits the console handler
    root.setLevel(cfg.logging.level)
    root.propagate = False
    if not cfg.logging.file_name:
        log.setup(cfg.logging.level)
    t0 = time.time()
    ok = run_one(cfg, *patient_args)
    return ok, time.time() - t0


def progress(done, total, failed, t0, width=30):
    """A progress bar line: [#####.....] 3/10 30% | 1 failed | 4.2 min, ETA 9.8 min."""
    filled = round(width * done / max(total, 1))
    elapsed = time.time() - t0
    eta = f"{elapsed / done * (total - done) / 60:.1f} min" if done else "?"
    return (f"[{'#' * filled}{'.' * (width - filled)}] {done}/{total} {100 * done / max(total, 1):.0f}%"
            f"{f' | {failed} failed' if failed else ''} | {elapsed / 60:.1f} min, ETA {eta}")


def run_sequential(cfg, patients, structures) -> list:
    """The patients one after the other; returns the names of those that failed."""
    failed, t0 = [], time.time()
    if len(patients) > 1:
        logger.info(progress(0, len(patients), 0, t0))
    for done, p in enumerate(patients, start=1):
        if not run_one(cfg, *p, structures):
            failed.append(p[0])
        if len(patients) > 1:
            logger.info(progress(done, len(patients), len(failed), t0))
    return failed


def run_parallel(cfg, patients, structures, jobs, heartbeat_s=60) -> list:
    """The patients on `jobs` processes; returns the names of those that failed. The progress
    line is repeated every `heartbeat_s` seconds while no patient finishes."""
    threads = max(1, usable_cpus() // jobs)
    logger.info(f"{len(patients)} patients, {jobs} at a time, {threads} threads each; each one logs to "
                f"its own {cfg.logging.file_name or 'console'}")
    failed, done, t0 = [], 0, time.time()
    logger.info(progress(0, len(patients), 0, t0))
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(_worker, cfg, (*p, structures), threads): p[0] for p in patients}
        pending = set(futures)
        while pending:
            finished, pending = wait(pending, timeout=heartbeat_s, return_when=FIRST_COMPLETED)
            if not finished:                           # still running: show that it is alive
                logger.info(progress(done, len(patients), len(failed), t0))
            for future in finished:
                patient = futures[future]
                try:
                    ok, seconds = future.result()
                except Exception as error:             # outside batch mode, or a crashed process
                    ok, seconds = False, 0.0
                    logger.error(f"patient {patient}: {error!r}")
                done += 1
                if not ok:
                    failed.append(patient)
                logger.info(f"{patient} {'done' if ok else 'FAILED'} ({seconds:.0f} s)")
                logger.info(progress(done, len(patients), len(failed), t0))
    return failed


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--structures", nargs="+", choices=cfgmod.STRUCTURE_NAMES,
                    help="only these structures (among those enabled in the config)")
    ap.add_argument("--jobs", type=int, default=1,
                    help="patients processed in parallel, one process each (default 1: one after the other)")
    a = ap.parse_args()
    cfg = cfgmod.load(a.config)
    log.setup(cfg.logging.level)

    structures = cfg.enabled_structures()
    if a.structures:
        structures = {s: c for s, c in structures.items() if s in a.structures}

    patients = list(iter_patients(cfg.batch))
    if not patients:
        logger.error(f"no patient in {cfg.batch.patients_root_dir}: a patient is a subfolder that "
                     f"contains the store {cfg.batch.store}")
        sys.exit(1)
    logger.info(f"{len(patients)} patient(s) in {cfg.batch.patients_root_dir}, structures {list(structures)}")
    if a.jobs > 1 and len(patients) > 1:
        failed = run_parallel(cfg, patients, structures, a.jobs)
    else:
        limit_threads(usable_cpus())
        failed = run_sequential(cfg, patients, structures)

    if failed:
        logger.error(f"failed patients ({len(failed)}): {', '.join(failed)}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
