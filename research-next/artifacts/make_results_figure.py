"""Static scientific figure from the audited supplemental run, no fitted model."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

PROJECT = Path(__file__).resolve().parents[1]
RUN = PROJECT / "runs/sensitivity-v1"
raw = (RUN / "summary.json").read_bytes()
summary = json.loads(raw)
audit = json.loads((RUN / "audit.json").read_text())
assert audit["status"] == "passed"
assert hashlib.sha256(raw).hexdigest() == audit["summary_sha256"]
models = ("HuggingFaceTB/SmolLM2-135M", "Qwen/Qwen2.5-0.5B")
fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.65), sharey=True)
colors = {"sparse": "#16849b", "all": "#c66c2f"}
for ax, model in zip(axes, models):
    info = summary["models"][model]
    strata = info["strata"]["layer"]
    layers = sorted(map(int, strata))
    n_layers = max(layers) + 1
    cuts = (n_layers // 4, n_layers // 2, 3*n_layers // 4)
    assert len(layers) == 3
    for method, offset, label in (("sparse", -.07, "Sparse RECUT"), ("all", .07, "All-layer journal")):
        q = [strata[str(layer)]["comparisons"]["checkpoint_alias_full"][method]["speedup"] for layer in layers]
        med = [a["median"] for a in q]
        low = [a["median"] - a["q25"] for a in q]
        high = [a["q75"] - a["median"] for a in q]
        ax.errorbar([i+offset for i in range(3)], med, yerr=[low, high],
                    color=colors[method], marker="o", markersize=6, capsize=4,
                    linewidth=1.7, label=label)
    ax.axhline(1, color="#555555", linestyle="--", linewidth=1)
    ax.set_yscale("log")
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"Layer {layer}\n(sparse cut {max([0]+[c for c in cuts if c <= layer])})" for layer in layers], fontsize=10)
    ax.set_title(model.split("/")[-1], loc="left", fontweight="bold", fontsize=12)
    ax.set_xlim(-.3, 2.3)
    ax.grid(axis="y", which="major", color="#dddddd", linewidth=.6)
    ax.spines[["top", "right"]].set_visible(False)
    counts = [strata[str(layer)]["n"] for layer in layers]
    ax.text(.02, .95, "Cases per layer: " + ", ".join(map(str, counts)), transform=ax.transAxes,
            va="top", fontsize=9, color="#555555")
axes[0].set_ylabel("Checkpoint-alias time / method time\n(log scale; above 1 = faster recovery)", fontsize=10)
axes[0].set_yticks([.5, 1, 2, 4, 8, 16, 32, 64])
axes[0].yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:g}x"))
all_q = [s["comparisons"]["checkpoint_alias_full"][m]["speedup"]
         for model in models for s in summary["models"][model]["strata"]["layer"].values()
         for m in ("sparse", "all")]
axes[0].set_ylim(min(.8, min(q["q25"] for q in all_q)*.85), max(2, max(q["q75"] for q in all_q)*1.45))
fig.suptitle("Observed recovery tradeoff by fault location", fontsize=15, fontweight="bold", x=.06, ha="left")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .045), ncol=2, frameon=False)
fig.text(.06, .02, "Points: median across paired case ratios. Bars: IQR across cases, not confidence intervals. Fixed workloads.", fontsize=8, color="#555555")
fig.subplots_adjust(left=.10, right=.98, top=.83, bottom=.26, wspace=.17)
out = PROJECT / "artifacts/figures/recovery_by_layer.png"
fig.savefig(out, dpi=180, facecolor="white")
plt.close(fig)
out.with_suffix('.json').write_text(json.dumps(dict(summary_sha256=hashlib.sha256(raw).hexdigest(),
    builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    image_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
    statistic='Across-case median and IQR of paired ratios of per-case median times; not confidence intervals'), indent=2)+'\n')
print(out)
