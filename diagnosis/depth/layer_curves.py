"""Final-epoch per-layer TEST accuracy (local readout) for every 16-layer method on CIFAR-100 and TinyIN."""
import csv, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
DATASETS = {"cifar100": "CIFAR-100", "tinyimagenet": "TinyIN"}
# fixed order -> fixed color + marker (secondary encoding)
METHODS = [("TTGS", "#2a78d6", "o"), ("TTG", "#eb6834", "s"), ("DTG-FF", "#1baf7a", "^"),
           ("HCL-FF", "#eda100", "D"), ("ASGE", "#e87ba4", "v"), ("CwC", "#008300", "P"),
           ("DeeperForward", "#6250d6", "X")]
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})

def load(dataset):
    data = {}
    for name, *_ in METHODS:
        row = json.loads((Path("results") / dataset / "16-depth" / name / "seed1" / "history.json").read_text())[-1]
        assert row["epoch"] == 400
        data[name] = [row[f"test/layer_{l}_accuracy"] for l in range(1, 17)]
    return data

def draw(ax, dataset, data):
    ends = []
    for name, color, marker in METHODS:
        y = data[name]
        ax.plot(range(1, 17), y, color=color, lw=2, marker=marker, ms=5, mec=SURFACE, mew=1, label=name, zorder=3)
        ends.append([y[-1], name, color])
    ends.sort()
    gap = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.045
    for i in range(1, len(ends)):  # de-collide end labels
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + gap)
    for y, name, _ in ends:
        ax.annotate(name, (16, y), xytext=(8, 0), textcoords="offset points", va="center", fontsize=8.5, color=INK)
    ax.set_title(f"{DATASETS[dataset]}, 16 layers", color=INK, loc="left", fontsize=11)
    ax.set_xlabel("Layer"); ax.set_ylabel("Test accuracy (%)")
    ax.set_xticks(range(1, 17)); ax.set_xlim(0.5, 18.6)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    for side in ("top", "right"): ax.spines[side].set_visible(False)

all_data = {d: load(d) for d in DATASETS}
with open(OUT / "layer_test_accuracy.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["dataset", "method"] + [f"L{l}" for l in range(1, 17)])
    for d, data in all_data.items():
        for name, *_ in METHODS: w.writerow([d, name] + [f"{v:.2f}" for v in data[name]])
for d, data in all_data.items():
    fig, ax = plt.subplots(figsize=(7.5, 5.0)); draw(ax, d, data)
    ax.set_title("", loc="left"); fig.suptitle(f"{DATASETS[d]}, 16 layers", x=0.02, ha="left", color=INK, fontsize=11)
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0.97), ncol=4,
               frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.90)); fig.savefig(OUT / f"layer_test_{d}_16.png", dpi=200); plt.close(fig)
fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
for ax, (d, data) in zip(axes, all_data.items()): draw(ax, d, data)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=7, frameon=False, fontsize=9)
fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(OUT / "layer_test_16_both.png", dpi=200)
print("ok")
