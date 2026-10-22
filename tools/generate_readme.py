#!/usr/bin/env python3
"""Regenerate the machine-owned sections of README.md from experiment frontmatter.

Usage:
    python tools/generate_readme.py            # rewrite README.md in place
    python tools/generate_readme.py --check    # exit 1 if regeneration would change it

README.md must contain these marker pairs. Everything between them is owned by
this script and will be overwritten. Everything outside is yours.

    <!-- BEGIN:experiments -->   <!-- END:experiments -->
    <!-- BEGIN:blocked -->       <!-- END:blocked -->
    <!-- BEGIN:roster -->        <!-- END:roster -->
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("PyYAML is required: pip install pyyaml")

REPO_ISSUES = "../../issues"

STATE_BADGE = {
    "proposed": "proposed",
    "scoping": "scoping",
    "smoke": "smoke",
    "ready": "ready",
    "queued": "queued",
    "running": "running",
    "analysis": "analysis",
    "writeup": "writeup",
    "done": "done",
    "blocked_human": "**blocked on Shreyas**",
    "killed": "killed",
}

SORT_KEY = {
    "blocked_human": 0, "running": 1, "queued": 2, "analysis": 3, "ready": 4,
    "smoke": 5, "writeup": 6, "scoping": 7, "proposed": 8, "done": 9, "killed": 10,
}


def read_frontmatter(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    parts = text.split("---", 2)
    if len(parts) < 3:
        raise ValueError(f"{path}: no frontmatter")
    data = yaml.safe_load(parts[1])
    if not isinstance(data, dict):
        raise ValueError(f"{path}: frontmatter is not a mapping")
    return data


def load_experiments(root: Path) -> list[dict]:
    exp_root = root / "experiments"
    if not exp_root.is_dir():
        return []
    out = []
    for folder in sorted(p for p in exp_root.iterdir() if p.is_dir()):
        f = folder / "EXPERIMENT.md"
        if f.exists():
            data = read_frontmatter(f)
            data["_path"] = f"experiments/{folder.name}"
            out.append(data)
    return sorted(out, key=lambda d: (SORT_KEY.get(d.get("state"), 99), d.get("id", "")))


def issue_link(num) -> str:
    return f"[#{num}]({REPO_ISSUES}/{num})" if num else ""


def experiments_table(exps: list[dict]) -> str:
    if not exps:
        return "_No experiments yet._"
    rows = ["| experiment | state | owner | issue | next action |",
            "| --- | --- | --- | --- | --- |"]
    for e in exps:
        next_action = (e.get("next_action") or "").strip().replace("\n", " ")
        if len(next_action) > 80:
            next_action = next_action[:77] + "..."
        rows.append(
            f"| [{e.get('title', e['id'])}]({e['_path']}/EXPERIMENT.md) "
            f"| {STATE_BADGE.get(e.get('state'), e.get('state'))} "
            f"| {e.get('owner', '')} "
            f"| {issue_link(e.get('issue'))} "
            f"| {next_action} |"
        )
    return "\n".join(rows)


def blocked_list(exps: list[dict]) -> str:
    blocked = [e for e in exps if e.get("state") == "blocked_human"]
    if not blocked:
        return "_Nothing is waiting on a decision._"
    lines = []
    for e in blocked:
        b = e.get("blocked_on") or {}
        lines.append(
            f"- **{e.get('title', e['id'])}** ({issue_link(e.get('issue'))}): "
            f"{(b.get('question') or '').strip()}  \n"
            f"  Recommendation: {b.get('recommendation')}  \n"
            f"  Defaults to *{b.get('default_action')}* on {b.get('deadline')}."
        )
    return "\n".join(lines)


def roster_table(root: Path) -> str:
    path = root / "roster.yaml"
    if not path.exists():
        return "_No roster.yaml._"
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    entries = data.get("roster", [])
    if not entries:
        return "_Roster is empty._"
    rows = ["| who | topic | status | notes |", "| --- | --- | --- | --- |"]
    for e in entries:
        rows.append(
            f"| {e.get('person') or '_unassigned_'} | {e.get('topic', '')} "
            f"| {e.get('status', '')} | {(e.get('notes') or '').strip()} |"
        )
    return "\n".join(rows)


def splice(text: str, name: str, body: str) -> str:
    begin, end = f"<!-- BEGIN:{name} -->", f"<!-- END:{name} -->"
    if begin not in text or end not in text:
        raise SystemExit(f"README.md is missing the {name} markers ({begin} ... {end})")
    head, rest = text.split(begin, 1)
    _, tail = rest.split(end, 1)
    return f"{head}{begin}\n{body}\n{end}{tail}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    readme = root / "README.md"
    original = readme.read_text(encoding="utf-8")
    exps = load_experiments(root)

    updated = splice(original, "experiments", experiments_table(exps))
    updated = splice(updated, "blocked", blocked_list(exps))
    updated = splice(updated, "roster", roster_table(root))

    if args.check:
        if updated != original:
            print("README.md is stale. Run: python tools/generate_readme.py")
            return 1
        print("README.md is up to date.")
        return 0

    readme.write_text(updated, encoding="utf-8")
    print(f"README.md regenerated from {len(exps)} experiment(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
