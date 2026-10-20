"""
BEA q-diagnostics: evolution + per-sample distribution.

Two plots are produced from a training log + final checkpoint:

1. ``q_evolution.png``    : per-epoch mean of q_source / q_target parsed from
                            the training log (two-line evolution plot,
                            source = blue, target = red).
2. ``q_distribution.png`` : per-sample histograms of q_source / q_target
                            extracted from the final checkpoint's BEA state
                            (densities overlaid, source = blue, target = red).

Usage:
    python scripts/plot_q_evolution.py \
        --log  runs/beacon_stack_sim/<run>/logs/log.txt \
        --ckpt runs/beacon_stack_sim/<run>/models/model_epoch_600.pth
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import MaxNLocator


SOURCE_COLOR = "#1f77b4"
TARGET_COLOR = "#d62728"

SOURCE_FILL = "#a8c7e6"
TARGET_FILL = "#f1a9a8"


def _lighten(hex_color: str, mix: float = 0.55) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    r = round(r + (255 - r) * mix)
    g = round(g + (255 - g) * mix)
    b = round(b + (255 - b) * mix)
    return f"#{r:02x}{g:02x}{b:02x}"


def parse_q_evolution(log_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract per-epoch (epoch, q_source_mean, q_target_mean) from a log file.

    Strategy: iterate the log line-by-line, detect ``Train Epoch <N>`` headers,
    and grab the JSON block that follows (delimited by '{' / '}').
    """
    text = log_path.read_text()

    epoch_re = re.compile(r"^Train Epoch\s+(\d+)\s*$", re.MULTILINE)

    epochs: list[int] = []
    q_src: list[float] = []
    q_tgt: list[float] = []

    for m in epoch_re.finditer(text):
        epoch = int(m.group(1))
        rest = text[m.end():]
        open_idx = rest.find("{")
        if open_idx == -1:
            continue
        depth = 0
        end_idx = -1
        for i, ch in enumerate(rest[open_idx:], start=open_idx):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end_idx = i + 1
                    break
        if end_idx == -1:
            continue
        block = rest[open_idx:end_idx]
        try:
            data = json.loads(block)
        except json.JSONDecodeError:
            continue
        if "q_source_mean" not in data or "q_target_mean" not in data:
            continue
        epochs.append(epoch)
        q_src.append(float(data["q_source_mean"]))
        q_tgt.append(float(data["q_target_mean"]))

    if not epochs:
        raise RuntimeError(f"No q_source_mean / q_target_mean entries found in {log_path}")

    return np.asarray(epochs), np.asarray(q_src), np.asarray(q_tgt)


def plot_q_evolution(
    epochs: np.ndarray,
    q_src: np.ndarray,
    q_tgt: np.ndarray,
    out_path: Path,
    title: str | None = None,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 14,
            "axes.titlesize": 18,
            "axes.titleweight": "normal",
            "axes.labelsize": 15,
            "axes.labelweight": "normal",
            "axes.edgecolor": "#333333",
            "axes.linewidth": 1.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.frameon": False,
            "legend.fontsize": 14,
        }
    )

    fig, ax = plt.subplots(figsize=(8.4, 5.2), dpi=160)
    ax.set_facecolor("#f5f5f7")
    ax.grid(True, linestyle="--", linewidth=0.7, color="white", zorder=0)
    ax.set_axisbelow(True)

    ax.fill_between(
        epochs,
        q_src,
        q_tgt,
        where=q_tgt >= q_src,
        color="#888888",
        alpha=0.10,
        linewidth=0,
        zorder=1,
        label=None,
    )

    ax.plot(
        epochs,
        q_src,
        color=SOURCE_COLOR,
        linewidth=2.2,
        zorder=3,
        label=r"source  $\bar q_{\rm src}$",
        solid_capstyle="round",
    )
    ax.plot(
        epochs,
        q_tgt,
        color=TARGET_COLOR,
        linewidth=2.2,
        zorder=3,
        label=r"target  $\bar q_{\rm tgt}$",
        solid_capstyle="round",
    )

    for x, y, color in [
        (epochs[-1], q_src[-1], SOURCE_COLOR),
        (epochs[-1], q_tgt[-1], TARGET_COLOR),
    ]:
        ax.scatter(
            [x], [y],
            s=45, color=color, edgecolor="white", linewidth=1.4, zorder=5,
        )
        ax.annotate(
            f"{y:.3f}",
            xy=(x, y),
            xytext=(8, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=12,
            color=color,
            fontweight="semibold",
        )

    ax.set_xlim(epochs.min(), epochs.max() * 1.06)
    ymin = min(q_src.min(), q_tgt.min())
    ymax = max(q_src.max(), q_tgt.max())
    pad = (ymax - ymin) * 0.08
    ax.set_ylim(ymin - pad, ymax + pad)

    ax.set_xlabel("Training epoch")
    ax.set_ylabel("Mean $q$")
    ax.set_title(title or "q-Evolution During Training")

    epoch_max = int(epochs.max())
    xticks = list(range(100, epoch_max + 1, 100))
    if xticks:
        ax.set_xticks(xticks)

    leg = ax.legend(loc="upper left", handlelength=2.4, borderaxespad=0.6)
    for line in leg.get_lines():
        line.set_linewidth(2.6)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def find_latest_checkpoint(run_dir: Path) -> Path | None:
    """Find the highest-epoch checkpoint under ``<run_dir>/models/``."""
    models_dir = run_dir / "models"
    if not models_dir.is_dir():
        return None
    epoch_re = re.compile(r"model_epoch_(\d+)")
    best_epoch = -1
    best_path: Path | None = None
    for p in models_dir.glob("model_epoch_*.pth"):
        m = epoch_re.search(p.name)
        if not m:
            continue
        ep = int(m.group(1))
        if ep > best_epoch:
            best_epoch = ep
            best_path = p
    return best_path


def load_q_from_ckpt(ckpt_path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    """Extract source/target per-sample q tensors and epoch from a checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    bea = ckpt["model"]["bea"]
    q = bea["q"].detach().cpu().numpy()
    m = int(bea["m"])
    n = int(bea["n"])
    epoch = int(ckpt.get("variable_state", {}).get("epoch", -1))
    return q[:m], q[m:m + n], epoch


def plot_q_distribution(
    q_src: np.ndarray,
    q_tgt: np.ndarray,
    out_path: Path,
    epoch: int | None = None,
    title: str | None = None,
    bins: int = 80,
) -> None:
    """Overlaid density histograms of per-sample q values for source/target."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 14,
            "axes.titlesize": 18,
            "axes.titleweight": "normal",
            "axes.labelsize": 15,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 1.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.frameon": False,
            "legend.fontsize": 13,
        }
    )

    fig, ax = plt.subplots(figsize=(8.4, 5.2), dpi=160)
    ax.set_facecolor("#f5f5f7")
    ax.grid(True, linestyle="--", linewidth=0.7, color="white", zorder=0)
    ax.set_axisbelow(True)

    lo = float(min(q_src.min(), q_tgt.min()))
    hi = float(max(q_src.max(), q_tgt.max()))
    pad = (hi - lo) * 0.02
    edges = np.linspace(lo - pad, hi + pad, bins + 1)

    src_label = (
        rf"source  (n={len(q_src):,},  $\mu$={q_src.mean():.3f}$\pm${q_src.std():.3f})"
    )
    tgt_label = (
        rf"target  (n={len(q_tgt):,},  $\mu$={q_tgt.mean():.3f}$\pm${q_tgt.std():.3f})"
    )

    ax.hist(
        q_src,
        bins=edges,
        density=True,
        color=SOURCE_FILL,
        edgecolor=SOURCE_COLOR,
        linewidth=1.0,
        alpha=0.75,
        zorder=2,
        label=src_label,
    )
    ax.hist(
        q_tgt,
        bins=edges,
        density=True,
        color=TARGET_FILL,
        edgecolor=TARGET_COLOR,
        linewidth=1.0,
        alpha=0.75,
        zorder=3,
        label=tgt_label,
    )

    for x, color in [(q_src.mean(), SOURCE_COLOR), (q_tgt.mean(), TARGET_COLOR)]:
        ax.axvline(x, color=color, linewidth=1.6, linestyle="--", alpha=0.85, zorder=4)

    ax.set_xlabel("Per-sample $q$")
    ax.set_ylabel("Normalized density")
    ax.set_title(title or "q-Distribution at The End of Training")

    ax.xaxis.set_major_locator(MaxNLocator(nbins=8))

    leg = ax.legend(
        loc="upper left",
        borderaxespad=0.6,
        frameon=True,
        framealpha=1.0,
        facecolor="white",
        edgecolor="none",
        borderpad=0.6,
    )
    leg.set_zorder(10)
    for patch in leg.get_patches():
        patch.set_alpha(0.9)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log",
        type=Path,
        required=True,
        help="Path to the training log file (<run_dir>/logs/log.txt).",
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=None,
        help=(
            "Optional checkpoint .pth file. If omitted, the latest "
            "model_epoch_*.pth under <run_dir>/models/ is used."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path for the evolution plot. Defaults to <log_dir>/q_evolution.png.",
    )
    parser.add_argument(
        "--dist_output",
        type=Path,
        default=None,
        help="Output PNG path for the distribution plot. Defaults to <log_dir>/q_distribution.png.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional title override for the evolution plot.",
    )
    parser.add_argument(
        "--no_dist",
        action="store_true",
        help="Skip the distribution plot.",
    )
    args = parser.parse_args()

    log_path = args.log.resolve()
    out_path = (args.output or log_path.parent / "q_evolution.png").resolve()

    print(f"[q-evolution] parsing {log_path}")
    epochs, q_src, q_tgt = parse_q_evolution(log_path)
    print(
        f"[q-evolution] parsed {len(epochs)} epochs "
        f"(epoch {epochs.min()}..{epochs.max()})"
    )

    plot_q_evolution(epochs, q_src, q_tgt, out_path, title=args.title)
    print(f"[q-evolution] wrote {out_path}")

    if args.no_dist:
        return

    run_dir = log_path.parent.parent
    ckpt_path = (args.ckpt or find_latest_checkpoint(run_dir))
    if ckpt_path is None:
        print(f"[q-distribution] no checkpoint found under {run_dir}/models/, skipping.")
        return
    ckpt_path = ckpt_path.resolve()
    dist_out = (args.dist_output or log_path.parent / "q_distribution.png").resolve()

    print(f"[q-distribution] loading {ckpt_path}")
    qs, qt, ep = load_q_from_ckpt(ckpt_path)
    print(
        f"[q-distribution] source: n={len(qs)}, mean={qs.mean():.4f}, std={qs.std():.4f}; "
        f"target: n={len(qt)}, mean={qt.mean():.4f}, std={qt.std():.4f}"
    )
    plot_q_distribution(qs, qt, dist_out, epoch=ep)
    print(f"[q-distribution] wrote {dist_out}")


if __name__ == "__main__":
    main()
