"""Check which dataset's stories actually reached the model in baseline / gamified logs.

The narrative (gt-harmbench-with-targets.csv) and gamified (gt-harmbench-gamify.csv)
datasets share payoffs, actions and the intro/closing paragraphs of every story, so
only the full story text tells them apart. For each sample this resolves the user
prompt of every model call and reports whether it contains the full narrative story,
the full gamified story, or neither, plus whether it states explicit "receive N points"
payoffs (present only in gamified prompts).

A clean narrative run reports normal/normal for every sample and no points; a clean
gamified run reports gamify/gamify and points everywhere. "none" means the prompt came
from a different dataset version (e.g. the early-January gamified runs, whose column
player saw the row player's payoffs).

Usage:
    uv run python3 scripts/verification/check_gamify_contamination.py logs/<run>.eval [...]
"""

import argparse
import csv
import json
import re
import zipfile
from collections import Counter
from pathlib import Path

csv.field_size_limit(10**9)
REPO = Path(__file__).resolve().parents[2]
POINTS = re.compile(r"receive -?\d+(\.\d+)? points")


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def load_stories(path: Path) -> dict[str, dict[str, str]]:
    return {
        r["id"]: {k: norm(r[k]) for k in ("story_row", "story_col")}
        for r in csv.DictReader(open(path))
    }


def resolve(content, attachments: dict) -> str:
    if not isinstance(content, str):
        content = " ".join(x.get("text", "") for x in content if isinstance(x, dict))
    if content.startswith("attachment://"):
        content = attachments.get(content[len("attachment://"):], "")
    return content


def label(prompt: str, sid: str, normal: dict, gamify: dict) -> str:
    """Label a prompt by which dataset's full row or column story it contains."""
    p = norm(prompt)
    n = any(normal[sid][k] in p for k in ("story_row", "story_col"))
    g = any(gamify[sid][k] in p for k in ("story_row", "story_col"))
    return {(True, False): "normal", (False, True): "gamify", (True, True): "both", (False, False): "none"}[(n, g)]


def check(path: Path, normal: dict, gamify: dict) -> None:
    z = zipfile.ZipFile(path)
    header = json.loads(z.read("header.json"))
    stories, points = Counter(), Counter()
    for name in z.namelist():
        if not name.startswith("samples/"):
            continue
        s = json.loads(z.read(name))
        sid = str(s["id"]).split("_")[0]
        att = s.get("attachments", {})
        labels = []
        for e in s["events"]:
            if e["event"] != "model":
                continue
            prompt = " ".join(resolve(m["content"], att) for m in e["input"] if m["role"] == "user")
            labels.append(label(prompt, sid, normal, gamify))
            points["points" if POINTS.search(prompt) else "no_points"] += 1
        stories["/".join(labels)] += 1
    print(f"\n== {path.name}")
    print(f"   model:   {header['eval']['model']}")
    print(f"   dataset: {header['eval']['dataset'].get('location', '?')}")
    print(f"   stories per sample (one label per model call): {dict(stories)}")
    print(f"   explicit payoffs per model call: {dict(points)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("logs", nargs="+", type=Path, help=".eval log files to check")
    parser.add_argument("--normal", type=Path, default=REPO / "data/gt-harmbench-with-targets.csv")
    parser.add_argument("--gamify", type=Path, default=REPO / "data/gt-harmbench-gamify.csv")
    args = parser.parse_args()
    normal, gamify = load_stories(args.normal), load_stories(args.gamify)
    for path in args.logs:
        check(path, normal, gamify)


if __name__ == "__main__":
    main()
