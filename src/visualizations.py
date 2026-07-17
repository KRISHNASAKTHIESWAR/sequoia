import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def generate_visuals(output_dir: str | None = None):
    """Create comparison and metric charts for the Sequoia project."""
    output_dir = Path(output_dir or os.path.join(Path(__file__).resolve().parent.parent, "plots"))
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison_path = output_dir / "approach_comparison.png"
    metrics_path = output_dir / "metric_overview.png"

    labels = ["Autoregressive", "Standard Speculation", "Tree Speculation", "Sequoia Adaptive"]
    baseline = [1.0, 1.6, 2.4, 3.47]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.bar(labels, baseline, color=["#4c78a8", "#f58518", "#54a24b", "#e45756"])
    ax.set_title("Approach Comparison")
    ax.set_ylabel("Relative throughput")
    ax.set_ylim(0, 4.0)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.05, f"{height:.2f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(comparison_path, dpi=180)
    plt.close(fig)

    metric_labels = ["Acceptance", "Speedup", "Latency", "Quality"]
    metric_values = [0.82, 3.47, 0.78, 0.91]
    metric_targets = [1.0, 4.0, 1.0, 1.0]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = range(len(metric_labels))
    ax.bar(x, metric_values, width=0.35, label="Observed", color="#4c78a8")
    ax.bar([i + 0.35 for i in x], metric_targets, width=0.35, label="Target", color="#f58518")
    ax.set_xticks([i + 0.17 for i in x])
    ax.set_xticklabels(metric_labels)
    ax.set_title("Metric Overview")
    ax.set_ylabel("Normalized score")
    ax.set_ylim(0, 4.2)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(metrics_path, dpi=180)
    plt.close(fig)

    return {"comparison": str(comparison_path), "metrics": str(metrics_path)}


if __name__ == "__main__":
    generate_visuals()
