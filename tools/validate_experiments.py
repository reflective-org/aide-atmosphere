#!/usr/bin/env python3
"""Validate experiment folders in jcm-experiments against the schema.

Usage:
    python tools/validate_experiments.py [--root .] [--strict]

Exits non-zero if any ERROR is found. --strict also fails on WARN.
The state gate table here is the authority; CLAUDE.md documents it for humans.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("PyYAML is required: pip install pyyaml")

KINDS = {"experiment", "tooling", "plan"}

# compute.machine must name a machine the runner has a skill for; a typo here
# leaves it unable to pick the submission and polling commands. See CLAUDE.md.
MACHINES = {"voltagepark"}

STATE_ORDER = [
    "proposed",
    "scoping",
    "smoke",
    "ready",
    "queued",
    "running",
    "analysis",
    "writeup",
    "done",
]
OFF_LADDER = {"blocked_human", "killed"}
STATES = set(STATE_ORDER) | OFF_LADDER

# Fields required to ENTER each state, cumulative up the ladder.
GATES: dict[str, list[str]] = {
    "proposed": ["id", "title", "kind", "owner", "state", "created", "updated"],
    "scoping": ["hypothesis"],
    "smoke": [
        "discriminating_diagnostic",
        "falsifier",
        "expected_effect.magnitude",
        "expected_effect.units",
        "adjacency.checked",
        "compute.machine",
        "compute.budget_gpu_hours",
        "smoke.config",
        "smoke.criteria",
    ],
    "ready": ["smoke.passed", "smoke.record"],
    "queued": [
        "noise_floor.estimate",
        "noise_floor.source",
        "noise_floor.ensemble_needed",
        "compute.per_run_cap_gpu_hours",
    ],
    "running": [],
    "analysis": [],
    "writeup": ["verdict", "verdict_basis"],
    "done": [],
}

BLOCKED_FIELDS = [
    "blocked_on.question",
    "blocked_on.options",
    "blocked_on.recommendation",
    "blocked_on.default_action",
    "blocked_on.deadline",
]

FORBIDDEN_SUFFIXES = {".log", ".npz", ".npy", ".nc", ".h5", ".hdf5", ".pkl", ".pickle", ".zarr"}
LARGE_FILE_BYTES = 1_000_000

MANIFEST_REQUIRED = {"job_id", "config_hash", "jcm_sha", "machine", "submitted"}
MANIFEST_ON_COMPLETION = {"exit_state", "gpu_hours", "scratch_path"}

VERDICTS = {"supported", "refuted", "inconclusive"}


class Report:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warns: list[str] = []

    def error(self, where: str, msg: str) -> None:
        self.errors.append(f"ERROR {where}: {msg}")

    def warn(self, where: str, msg: str) -> None:
        self.warns.append(f"WARN  {where}: {msg}")


def get(data: dict, dotted: str):
    """Fetch a dotted path, returning None if any level is missing."""
    cur = data
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def is_empty(value) -> bool:
    return value is None or value == "" or value == [] or value == {}


def read_frontmatter(path: Path, rep: Report) -> dict | None:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        rep.error(str(path), "no YAML frontmatter block")
        return None
    parts = text.split("---", 2)
    if len(parts) < 3:
        rep.error(str(path), "frontmatter block is not closed")
        return None
    try:
        data = yaml.safe_load(parts[1])
    except yaml.YAMLError as exc:
        rep.error(str(path), f"frontmatter does not parse: {exc}")
        return None
    if not isinstance(data, dict):
        rep.error(str(path), "frontmatter is not a mapping")
        return None
    return data


def required_upto(state: str) -> list[str]:
    if state in OFF_LADDER:
        return list(GATES["proposed"])
    fields: list[str] = []
    for s in STATE_ORDER:
        fields.extend(GATES[s])
        if s == state:
            break
    return fields


def as_date(value):
    if isinstance(value, dt.date):
        return value
    if isinstance(value, str):
        try:
            return dt.date.fromisoformat(value[:10])
        except ValueError:
            return None
    return None


def check_manifest(
    folder: Path, state: str, rep: Report, retrofitted: bool = False
) -> tuple[int, set[str]]:
    """Validate runs/manifest.jsonl; return (entry count, set of job_ids)."""
    manifest = folder / "runs" / "manifest.jsonl"
    where = str(manifest)
    # A manifest is required from `ready` onward: entering ready means a smoke
    # run was submitted and recorded (smoke.record must name one of its jobs).
    needs_manifest = {"ready", "queued", "running", "analysis", "writeup", "done"}
    if not manifest.exists():
        if state in needs_manifest and not retrofitted:
            rep.error(where, f"state is {state} but no manifest exists")
        return 0, set()
    # The manifest is append only: a completing job is recorded by appending a
    # record carrying the same job_id plus the completion fields, never by
    # rewriting the submission line. Fold records by job_id (later wins, empty
    # values never overwrite) so a job is judged on its accumulated state.
    folded: dict[str, dict] = {}
    last_line: dict[str, int] = {}
    for lineno, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            rep.error(where, f"line {lineno} is not valid JSON: {exc}")
            continue
        if not isinstance(entry, dict):
            rep.error(where, f"line {lineno} is not a JSON object")
            continue
        if is_empty(entry.get("job_id")):
            rep.error(where, f"line {lineno} has no job_id, it cannot be folded onto a job")
            continue
        job_id = str(entry["job_id"])
        merged = folded.setdefault(job_id, {})
        for key, value in entry.items():
            if not is_empty(value):
                merged[key] = value
        last_line[job_id] = lineno

    n_open = 0
    for job_id, entry in folded.items():
        lineno = last_line[job_id]
        missing = sorted(f for f in MANIFEST_REQUIRED if is_empty(entry.get(f)))
        if missing:
            rep.error(where, f"job {job_id} (through line {lineno}) missing {missing}")
        if is_empty(entry.get("exit_state")):
            n_open += 1
        else:
            # A completed job with no scratch_path strands the analyst, and no
            # gpu_hours loses the accounting: errors, not warnings.
            missing_done = sorted(f for f in MANIFEST_ON_COMPLETION if is_empty(entry.get(f)))
            if missing_done:
                rep.error(
                    where, f"job {job_id} (through line {lineno}) completed but missing {missing_done}"
                )
    if state in {"analysis", "writeup", "done"} and n_open and not retrofitted:
        rep.error(where, f"state is {state} but {n_open} job(s) have no exit_state")
    if state in needs_manifest and not folded and not retrofitted:
        rep.error(where, f"state is {state} but manifest is empty")
    return len(folded), set(folded)


def check_artifacts(folder: Path, rep: Report, declared: list[str] | None = None) -> None:
    """Enforce the run-output ban, minus the folder's declared deliverables.

    `declared` comes from the `artifacts:` frontmatter list: paths relative to
    the folder, a directory entry covering everything under it. A declared
    binary deliverable (a trained model, a lookup table) is the same exception
    figures already are — it is the result, not run output — so it escapes the
    suffix ban but not the size warning.
    """
    declared = [d.rstrip("/") for d in (declared or [])]
    matched: set[str] = set()

    def is_declared(rel: str) -> bool:
        hit = False
        for d in declared:
            if rel == d or rel.startswith(d + "/"):
                matched.add(d)
                hit = True
        return hit

    for path in folder.rglob("*"):
        if not path.is_file():
            continue
        exempt = is_declared(path.relative_to(folder).as_posix())
        if path.suffix.lower() in FORBIDDEN_SUFFIXES and not exempt:
            rep.error(str(path), "run output must live in scratch and be referenced by the manifest")
        elif path.stat().st_size > LARGE_FILE_BYTES:
            rep.warn(str(path), f"{path.stat().st_size // 1000} kB tracked in git")

    for d in declared:
        if d not in matched:
            rep.warn(str(folder), f"artifacts entry {d!r} matches no file in the folder")


def check_experiment(folder: Path, roster_ids: set[str], rep: Report) -> dict | None:
    exp_file = folder / "EXPERIMENT.md"
    if not exp_file.exists():
        rep.error(str(folder), "no EXPERIMENT.md")
        return None
    data = read_frontmatter(exp_file, rep)
    if data is None:
        return None
    where = str(exp_file)

    if data.get("id") != folder.name:
        rep.error(where, f"id {data.get('id')!r} does not match directory name {folder.name!r}")
    if data.get("kind") not in KINDS:
        rep.error(where, f"kind must be one of {sorted(KINDS)}")

    state = data.get("state")
    if state not in STATES:
        rep.error(where, f"state {state!r} is not a known state")
        return data

    # Historical folders predate the schema. They are recorded, not preregistered.
    # retrofitted: true relaxes the prereg gates but not the artifact rules.
    retrofitted = data.get("retrofitted") is True
    if retrofitted and state not in {"done", "killed", "proposed", "blocked_human"}:
        rep.error(where, "retrofitted experiments must be done, killed, proposed or blocked_human")

    for field in (GATES["proposed"] if retrofitted else required_upto(state)):
        if is_empty(get(data, field)):
            rep.error(where, f"state is {state} but {field} is empty")

    def reached(gate_state: str) -> bool:
        return (
            not retrofitted
            and state in STATE_ORDER
            and STATE_ORDER.index(state) >= STATE_ORDER.index(gate_state)
        )

    # smoke.passed must be strictly true from `ready` onward, not just at
    # `ready` itself — otherwise an experiment can skip straight to a compute
    # state with smoke.passed: false (non-empty, so the gate loop misses it).
    if reached("ready") and get(data, "smoke.passed") is not True:
        rep.error(where, f"state is {state} but smoke.passed is not true")
    if state == "blocked_human":
        for field in BLOCKED_FIELDS:
            if is_empty(get(data, field)):
                rep.error(where, f"blocked_human requires {field}")
        deadline = as_date(get(data, "blocked_on.deadline"))
        if deadline is None:
            rep.error(where, "blocked_on.deadline is not a date")
        elif deadline < dt.date.today():
            rep.warn(where, f"blocked since {deadline}, past deadline, default_action is due")
    if state == "killed" and is_empty(data.get("kill_reason")):
        rep.error(where, "killed requires kill_reason")
    if not is_empty(data.get("verdict")) and data.get("verdict") not in VERDICTS:
        rep.error(where, f"verdict must be one of {sorted(VERDICTS)}")
    machine = get(data, "compute.machine")
    if not is_empty(machine) and machine not in MACHINES:
        rep.error(where, f"compute.machine {machine!r} must be one of {sorted(MACHINES)}")
    if state == "done" and not retrofitted:
        draft = folder / "draft.md"
        if not draft.exists() or not draft.read_text(encoding="utf-8").strip():
            rep.error(where, "state is done but draft.md is missing or empty")

    past_smoke_gate = (
        not retrofitted
        and state in STATE_ORDER
        and STATE_ORDER.index(state) >= STATE_ORDER.index("smoke")
    )
    if past_smoke_gate:
        prior = [p for p in (data.get("prior_work") or []) if isinstance(p, dict)]
        if not prior or all(is_empty(p.get("delta")) for p in prior):
            rep.error(where, "no prior_work entry has an articulated delta")

    # roster.yaml is a gate: a recorded conflict means coordinate first. An
    # experiment carrying one may not sit on the ladder, least of all in a
    # compute state — it belongs in blocked_human until the overlap is resolved.
    for conflict in (get(data, "adjacency.conflicts") or []):
        if conflict not in roster_ids:
            rep.warn(where, f"adjacency conflict {conflict!r} is not in roster.yaml")
        if state not in OFF_LADDER:
            rep.error(
                where,
                f"adjacency conflict with {conflict!r} but state is {state}, not blocked_human",
            )

    updated = as_date(data.get("updated"))
    if updated is None:
        rep.error(where, "updated is not a date")
    elif state in {"queued", "running"} and (dt.date.today() - updated).days > 14:
        rep.warn(where, f"state is {state} but nothing written for {(dt.date.today() - updated).days} days")

    if state not in {"done", "killed"} and is_empty(data.get("next_action")):
        rep.error(where, "next_action is empty, the next session will not know what to do")

    if not (folder / "LOG.md").exists():
        rep.warn(str(folder), "no LOG.md")

    _, job_ids = check_manifest(folder, state, rep, retrofitted)

    # From `ready` onward the claimed smoke run must actually exist: the
    # config file on disk, and the record resolving to a manifest job_id.
    # (Config-hash equivalence at submission time is the pre-submission
    # hook's job; the validator establishes the referenced things are real.)
    if reached("ready"):
        cfg = get(data, "smoke.config")
        if not is_empty(cfg):
            configs_dir = (folder / "configs").resolve()
            # Resolve before comparing: an absolute path discards `folder` and
            # a path with ../ escapes it, either of which would let an
            # unrelated file stand in for the immutable smoke config.
            resolved = (folder / str(cfg)).resolve()
            if not resolved.is_relative_to(configs_dir):
                rep.error(where, f"smoke.config {cfg!r} is outside {folder.name}/configs/")
            elif not resolved.is_file():
                rep.error(where, f"smoke.config {cfg!r} does not exist under {folder.name}/")
        record = get(data, "smoke.record")
        if not is_empty(record) and str(record) not in job_ids:
            rep.error(where, f"smoke.record {record!r} matches no job_id in runs/manifest.jsonl")

    declared = data.get("artifacts") or []
    if not isinstance(declared, list) or not all(isinstance(d, str) and d for d in declared):
        rep.error(where, "artifacts must be a list of non-empty paths relative to the folder")
        declared = []
    check_artifacts(folder, rep, declared)
    return data


def check_roster(root: Path, rep: Report) -> set[str]:
    path = root / "roster.yaml"
    if not path.exists():
        rep.error(str(path), "roster.yaml is missing, the adjacency gate cannot run")
        return set()
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        rep.error(str(path), f"does not parse: {exc}")
        return set()
    ids = set()
    for entry in data.get("roster", []):
        if not isinstance(entry, dict):
            rep.error(str(path), f"roster entry is not a mapping: {entry!r}")
            continue
        for field in ("id", "topic", "status"):
            if is_empty(entry.get(field)):
                rep.error(str(path), f"entry {entry.get('id', '?')!r} missing {field}")
        if is_empty(entry.get("person")) and entry.get("status") != "unassigned":
            rep.error(str(path), f"entry {entry.get('id', '?')!r} has no person but is not unassigned")
        if entry.get("id"):
            ids.add(entry["id"])
    return ids


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".", type=Path)
    parser.add_argument("--strict", action="store_true", help="fail on warnings too")
    args = parser.parse_args()

    rep = Report()
    root = args.root.resolve()
    roster_ids = check_roster(root, rep)

    exp_root = root / "experiments"
    if not exp_root.is_dir():
        rep.error(str(exp_root), "experiments/ does not exist")
        folders = []
    else:
        folders = sorted(p for p in exp_root.iterdir() if p.is_dir() and not p.name.startswith("."))

    for folder in folders:
        check_experiment(folder, roster_ids, rep)

    for line in rep.errors + rep.warns:
        print(line)
    print(f"\n{len(folders)} experiment(s), {len(rep.errors)} error(s), {len(rep.warns)} warning(s)")

    if rep.errors:
        return 1
    if args.strict and rep.warns:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
