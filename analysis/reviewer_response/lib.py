"""Shared, deterministic helpers for the reviewer-response analysis.

Read-only: loads the clean benchmark instances and the extracted model choices,
derives the ordinal quantities used across Tasks 1-5, and reuses the repo's own
structural verifier (`is_valid_game`) as the payoff-derived classifier.

Notation (per the reviewer's brief), defined generally so it also covers BoS,
whose diagonal cells are within-cell *unequal*:
  cell totals  t11,t12,t21,t22  (t12==t21 by mirror symmetry)
  diagonal payoff values  = {c11[0],c11[1],c22[0],c22[1]},  M_val = max
  off-diagonal payoff vals = {c12[0],c12[1],c21[0],c21[1]}, T = max, S = min
  For textbook-symmetric games this reduces exactly to D1,D2 scalars and the
  two off-diagonal values, so {2*D1, 2*D2, S+T} == {t11, t22, t_off}.
"""
from __future__ import annotations

import ast
import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.generation.fix_contextualizations import is_valid_game  # noqa: E402

CLEAN_CSV = REPO / "data" / "gt-harmbench-cleaned.csv"
CHOICES_CSV = Path(__file__).resolve().parent / "data" / "model_choices.csv"
CELL_KEYS = ["1_1_payoff", "1_2_payoff", "2_1_payoff", "2_2_payoff"]
GAME_TYPES = ["Prisoner's Dilemma", "Chicken", "Stag hunt", "Coordination",
              "Bach or Stravinski", "No conflict", "Matching pennies"]
# cell index order used everywhere: 0=(1,1) 1=(1,2) 2=(2,1) 3=(2,2)
CELL_ORDER = [(0, 0), (0, 1), (1, 0), (1, 1)]


@dataclass
class Instance:
    id: str
    game: str                    # stored formal_game label
    cells: list                  # [c11, c12, c21, c22], each [row, col]
    actions_row: list
    actions_col: list
    derived: list = field(default_factory=list)   # classes whose verifier passes

    @property
    def totals(self):
        return [c[0] + c[1] for c in self.cells]   # [t11, t12, t21, t22]

    @property
    def diag_vals(self):
        return [self.cells[0][0], self.cells[0][1], self.cells[3][0], self.cells[3][1]]

    @property
    def offdiag_vals(self):
        return [self.cells[1][0], self.cells[1][1], self.cells[2][0], self.cells[2][1]]

    @property
    def M_val(self):
        return max(self.diag_vals)

    @property
    def T(self):
        return max(self.offdiag_vals)

    @property
    def S(self):
        return min(self.offdiag_vals)

    @property
    def cond_TMS(self):
        """T > M > S : the only regime where the utilitarian argmax is NOT
        pinned by the ordinal ranking alone."""
        return self.T > self.M_val > self.S

    @property
    def D1(self):  # diagonal (1,1) representative (row payoff); scalar when symmetric
        return self.cells[0][0]

    @property
    def D2(self):
        return self.cells[3][0]

    def payoffs_dict(self):
        return {k: self.cells[i] for i, k in enumerate(CELL_KEYS)}


def load_instances() -> list[Instance]:
    rows = list(csv.DictReader(open(CLEAN_CSV)))
    insts = []
    for r in rows:
        cells = [ast.literal_eval(r[k]) for k in CELL_KEYS]
        inst = Instance(
            id=r["id"], game=r["formal_game"], cells=cells,
            actions_row=ast.literal_eval(r["actions_row"]),
            actions_col=ast.literal_eval(r["actions_column"]),
        )
        pd = inst.payoffs_dict()
        inst.derived = [g for g in GAME_TYPES if _safe_valid(g, pd)]
        insts.append(inst)
    return insts


def _safe_valid(game_type, payoffs) -> bool:
    try:
        return bool(is_valid_game(game_type, payoffs))
    except Exception:
        return False


def load_choices() -> dict:
    """Return {model: {instance_id: cell_index}} for status=='ok', deduped.

    cell_index in 0..3 matching CELL_ORDER. If a (model,instance) has multiple
    ok rows that disagree, it is dropped (recorded in load_choices.dropped)."""
    rows = list(csv.DictReader(open(CHOICES_CSV)))
    per: dict = {}
    seen: dict = {}
    dropped = 0
    for r in rows:
        if r["status"] != "ok":
            continue
        cell = CELL_ORDER.index((int(r["row_idx"]), int(r["col_idx"])))
        key = (r["model"], r["instance_id"])
        if key in seen:
            if seen[key] != cell:
                per.setdefault(r["model"], {}).pop(r["instance_id"], None)
                dropped += 1
            continue
        seen[key] = cell
        per.setdefault(r["model"], {})[r["instance_id"]] = cell
    load_choices.dropped = dropped
    return per


def util_argmax_set(totals) -> frozenset:
    mx = max(totals)
    return frozenset(i for i, t in enumerate(totals) if t == mx)
