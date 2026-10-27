
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =========================================================
# Replace these with your actual best-result runs/folds
# Each list = repeated runs / CV folds / seeds
# =========================================================
metrics_data = {
    "Accuracy": [0.81, 0.83, 0.81, 0.81],
    "F1 score": [0.78, 0.80, 0.81, 0.80],
    "AUROC":    [0.82, 0.83, 0.82, 0.83, 0.82],
    "Kappa": [0.78, 0.77, 0.79, 0.78, 0.79]
}

# =========================================================
# Output folder
# =========================================================
out_dir = Path("/home/ubuntu/SEEg results /Sozlocalization results ")
out_dir.mkdir(parents=True, exist_ok=True)

# =========================================================
# Nature-like style
# =========================================================
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# =========================================================
# Prepare values
# =========================================================
metric_names = list(metrics_data.keys())
means = [np.mean(metrics_data[m]) for m in metric_names]

# =========================================================
# Figure
# =========================================================
fig, ax = plt.subplots(figsize=(2.0, 2.3))

x = np.arange(len(metric_names))
bar_width = 0.82

# Blue bar color
bar_color = "#4C78A8"

bars = ax.bar(
    x,
    means,
    width=bar_width,
    color=bar_color,
    edgecolor=bar_color,
    linewidth=0.8,
    zorder=2
)

# =========================================================
# Overlay individual run points
# =========================================================
rng = np.random.default_rng(42)

for i, m in enumerate(metric_names):
    vals = np.array(metrics_data[m], dtype=float)
    jitter = rng.uniform(-0.12, 0.12, size=len(vals))

    ax.scatter(
        np.full(len(vals), x[i]) + jitter,
        vals,
        s=10,
        facecolors="none",
        edgecolors="#9e9e9e",
        linewidths=0.7,
        zorder=3
    )

# =========================================================
# Axes formatting
# =========================================================
ax.set_xticks(x)
ax.set_xticklabels(metric_names, rotation=45, ha="right", rotation_mode="anchor")

all_vals = np.concatenate([np.array(v) for v in metrics_data.values()])
ymin = max(0.0, np.floor((all_vals.min() - 0.03) * 10) / 10)
ymax = 0.9

ax.set_ylim(ymin, ymax)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.yaxis.grid(True, linestyle="-", linewidth=0.4, alpha=0.25, zorder=1)
ax.set_axisbelow(True)

ax.set_ylabel("")

plt.tight_layout()

png_path = out_dir / "private dataset1.png"
tiff_path = out_dir / "nature_private dataset2.tiff"
pdf_path = out_dir / "dataset3.pdf"

fig.savefig(png_path, dpi=600, bbox_inches="tight")
fig.savefig(tiff_path, dpi=600, bbox_inches="tight")
fig.savefig(pdf_path, bbox_inches="tight")

print("Saved:")
print(tiff_path)
print(pdf_path)

plt.show()

