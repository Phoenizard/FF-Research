"""Depth degradation per method on CIFAR-100, seed 1, from history.json only. Run from the repo root:
python -m diagnosis.depth.degradation

Criteria fixed on 2026-11-06 before any 16-layer run. Window W = epochs 381-400; every quantity is the W mean.
- F / Ftr: test / train fusion accuracy. a_l / b_l: test / train local accuracy of layer l.
- Drop = max_l a_l - a_D; TDrop = max_l b_l - b_D (D = depth, peak ties -> smaller l).
- dF = F(16) - F(8); dFtr = Ftr(16) - Ftr(8).
- Class: E1 if Drop16 >= 2.0, E2 if dF <= -1.0 (E = E1 or E2); otherwise N if dF >= +1.0, else S.
- Cause (E only): with E1, C1 (underfitting) if TDrop16 >= 2.0, C2 (generalization) if TDrop16 <= 0.5, else C3;
  with only E2, C1 if dFtr <= -1.0, C2 if dFtr >= 0.0, else C3.
- 8-layer reference: same method; TTG-8 for TTGS and its ablations (no TTGS-8 run yet).
"""
import json
from pathlib import Path

METHODS = ["ASGE", "DTG-FF", "CwC", "DeeperForward", "HCL-FF", "TTG", "TTG-Skip", "TTGS-Expected", "TTGS"]
REF8 = {"TTG-Skip": "TTG", "TTGS-Expected": "TTG", "TTGS": "TTG"}
W = range(381, 401)


def load(depth, method):
    path = Path(f"results/cifar100/{depth}-depth/{method}/seed1/history.json")
    rows = {r["epoch"]: r for r in json.loads(path.read_text())}
    assert sorted(rows) == list(range(1, 401)), path
    mean = lambda key: sum(rows[e][key] for e in W) / len(W)
    a = [mean(f"test/layer_{l}_accuracy") for l in range(1, depth + 1)]
    b = [mean(f"train/layer_{l}_accuracy") for l in range(1, depth + 1)]
    peak = max(range(depth), key=lambda l: (a[l], -l))
    train_peak = max(range(depth), key=lambda l: (b[l], -l))
    return {"F": mean("test/fusion_accuracy"), "Ftr": mean("train/fusion_accuracy"), "l_star": peak + 1,
            "Drop": a[peak] - a[-1], "lt_star": train_peak + 1, "TDrop": b[train_peak] - b[-1], "a": a, "b": b}


def classify(r16, r8):
    d_f, d_ftr = r16["F"] - r8["F"], r16["Ftr"] - r8["Ftr"]
    e1, e2 = r16["Drop"] >= 2.0, d_f <= -1.0
    if not (e1 or e2):
        return {"dF": d_f, "dFtr": d_ftr, "class": "N" if d_f >= 1.0 else "S", "cause": None}
    measure, high, low = (r16["TDrop"], 2.0, 0.5) if e1 else (-d_ftr, 1.0, 0.0)
    cause = "C1" if measure >= high else "C2" if measure <= low else "C3"
    return {"dF": d_f, "dFtr": d_ftr, "class": "E" + "1" * e1 + "2" * e2, "cause": cause}


def main():
    results = {}
    for method in METHODS:
        r8, r16 = load(8, REF8.get(method, method)), load(16, method)
        results[method] = {"ref8": REF8.get(method, method), "8": r8, "16": r16, **classify(r16, r8)}
        x = results[method]
        print(f"{method:14s} dF {x['dF']:+.2f} dFtr {x['dFtr']:+.2f} F16 {r16['F']:.2f} Drop16 {r16['Drop']:.2f} "
              f"TDrop16 {r16['TDrop']:.2f} {x['class']} {x['cause']}")
    Path(__file__).with_name("degradation.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
