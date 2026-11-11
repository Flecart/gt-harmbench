"""Build logs_manifest.csv: every local .eval log, what it is, and where its Drive copy is.

Temporary bookkeeping for the camera-ready (logs/ is gitignored; this CSV is the
index). Drive copies are matched by md5, so renamed files still match.

  PYTHONPATH=. uv run python -m analysis.reviewer_response.build_log_manifest
"""
from __future__ import annotations

import csv, hashlib, json, subprocess
from pathlib import Path

from inspect_ai.log import read_eval_log

from analysis.reviewer_response.reproduce_tables import LOGS

REPO = Path(__file__).resolve().parents[2]
LOG_ROOT = REPO / "logs"
OUT = Path(__file__).resolve().parent / "logs_manifest.csv"
DRIVE_ROOT = "1hyXR_iLqhZiUV8VdqUeI5eMZ6_nA_1iN"  # shared "GT-HarmBench logs" folder
PAPER_MODELS = list(LOGS)[:15]

# role of each local folder; refined per file in role()
FOLDER_ROLE = {
    "baseline": "RQ1 Table 2 / Tables 6-8",
    "abstract": "RQ2 abstract framing",
    "RQ2 - gamified": "RQ2 gamified framing",
    "variance": "reviewer response: k=5 variance",
    "Mech-Design": "RQ4 mechanisms",
    "reasoning": "RQ3 reasoning",
    "rq2-narrative": "RQ2",
    "baseline-superseded": "superseded (not in paper)",
    "broken": "broken (not in paper)",
    "Legacy": "legacy (not in paper)",
    "GDrive": "mirror of Drive RQ1 - baseline",
}


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def drive_index() -> dict[str, str]:
    out = subprocess.run(["rclone", "lsjson", "-R", "--hash", "--files-only",
                          "--drive-root-folder-id", DRIVE_ROOT, "gdrive:"],
                         capture_output=True, text=True, check=True).stdout
    idx = {}
    for f in json.loads(out):
        h = (f.get("Hashes") or {}).get("md5")
        if h:
            idx.setdefault(h, f["Path"])
    return idx


def role(rel: Path, task: str, dataset: str) -> str:
    top = rel.parts[0]
    r = FOLDER_ROLE.get(top, "unknown")
    paper = {str(p.relative_to(LOG_ROOT)): m for m, p in LOGS.items()}
    if str(rel) in paper:
        m = paper[str(rel)]
        return f"PAPER RQ1 Table 2 ({m})" if m in PAPER_MODELS else f"not in paper ({m})"
    if top == "baseline":
        return "not in paper"
    if top == "abstract" and "rq2-abstract" not in task:
        return "pilot (not in paper)"
    return r


def main():
    drive = drive_index()
    rows = []
    for p in sorted(LOG_ROOT.rglob("*.eval")):
        rel = p.relative_to(LOG_ROOT)
        try:
            h = read_eval_log(str(p), header_only=True)
            meta = dict(model=h.eval.model, task=h.eval.task, dataset=h.eval.dataset.name,
                        n=h.eval.dataset.samples, created=h.eval.created, status=h.status,
                        machine=str(h.eval.dataset.location or "").split("/gt-harmbench")[0])
        except Exception as e:  # unreadable log
            meta = dict(model="", task="", dataset="", n="", created="", status=f"unreadable: {e.__class__.__name__}", machine="")
        digest = md5(p)
        rows.append(dict(local_path=str(rel), role=role(rel, meta["task"], meta["dataset"]), **meta,
                         size_mb=round(p.stat().st_size / 1e6, 1), md5=digest,
                         drive_path=drive.get(digest, "")))
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} logs -> {OUT.relative_to(REPO)}; on Drive: {sum(bool(r['drive_path']) for r in rows)}")


if __name__ == "__main__":
    main()
