"""Turns the CSV files in results/ into graphs and a summary for the README.

Usage (from anywhere, after running the C++/CUDA program):
    py -m pip install numpy matplotlib
    py viewer\\plots.py

Writes into results/plots/:
    time_vs_size.png        RQ1  time of each version at each world size
    speedup_vs_size.png     RQ1  how many times faster the GPU is (copies included)
    gpu_time_breakdown.png  RQ4  share of GPU time spent copying vs. computing
    octaves.png             RQ2  time and speedup for 1 to 8 octaves
    blocks.png              RQ3  kernel time for each threads-per-block setting
    heightmaps.png          the three terrain types side by side
    <type>_1024x1024.png    one small picture per terrain type
and results/summary.md: the same numbers as ready-to-paste Markdown tables.

Not part of any timing. A graph is skipped if its CSV file is missing.
"""
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RES = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results"
OUT = RES / "plots"
OUT.mkdir(parents=True, exist_ok=True)

# Colors: fixed order, one per version. Text stays dark; only the marks carry color.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SERIES = [  # (label, color, marker)
    ("CPU, 1 core", BLUE, "o"),
    ("CPU, all cores", ORANGE, "s"),
    ("GPU total (with copies)", AQUA, "^"),
    ("GPU kernel only", YELLOW, "D"),
]
TYPE_COLOR = {"plains": BLUE, "hilly": ORANGE, "mountains": AQUA}
TYPE_MARK = {"plains": "o", "hilly": "s", "mountains": "^"}
TYPES = ["plains", "hilly", "mountains"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.axisbelow": True, "lines.linewidth": 2, "lines.markersize": 6, "legend.frameon": False,
})


def load(name):
    path = RES / name
    if not path.exists():
        print(f"  (skipped: {name} not found)")
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def num(row, key):
    v = row.get(key, "")
    return float(v) if v not in ("", None) else float("nan")


def save(fig, name):
    fig.savefig(OUT / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved plots/{name}")


def ms(v):
    """Short time text: 0.0036 ms, 0.98 ms, 15.5 ms, 1578 ms."""
    if v != v:
        return "-"
    if v < 0.1:
        return f"{v:.4f}"
    if v < 10:
        return f"{v:.2f}"
    if v < 100:
        return f"{v:.1f}"
    return f"{v:.0f}"


def by_version(rows, **match):
    """Rows that match the filters, split into cpu_1core / cpu_all_cores / GPU."""
    out = {"cpu_1core": [], "cpu_all_cores": [], "gpu": []}
    for r in rows:
        if all(str(r[k]) == str(v) for k, v in match.items()):
            out["gpu" if r["version"].startswith("gpu") else r["version"]].append(r)
    return out


summary = ["# Results summary",
           "",
           "Made by `viewer/plots.py` from the CSV files in this folder. "
           "Times are medians in milliseconds.", ""]
env = RES / "environment.txt"
if env.exists():
    summary += ["## Environment", "", "```text", env.read_text().strip(), "```", ""]

# ---------------------------------------------------------------------------
# RQ1 / RQ4: world size
# ---------------------------------------------------------------------------
sizes = load("benchmark_sizes.csv")
if sizes:
    present = [t for t in TYPES if any(r["type"] == t for r in sizes)]

    # --- time vs size, one panel per terrain type ---
    fig, axes = plt.subplots(1, len(present), figsize=(5.2 * len(present), 4.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, t in zip(axes, present):
        v = by_version(sizes, type=t)
        lines = [
            (v["cpu_1core"], "median_ms"), (v["cpu_all_cores"], "median_ms"),
            (v["gpu"], "median_ms"), (v["gpu"], "kernel_median_ms"),
        ]
        for (rows, key), (label, color, marker) in zip(lines, SERIES):
            if rows:
                ax.plot([num(r, "width") for r in rows], [num(r, key) for r in rows],
                        color=color, marker=marker, label=label, markeredgecolor=SURFACE)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ticks = sorted({num(r, "width") for r in v["cpu_1core"]})
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{int(w)}" for w in ticks], rotation=45)
        ax.minorticks_off()
        ax.set_title(f"{t} ({v['cpu_1core'][0]['octaves']} octaves)")
        ax.set_xlabel("world side length (columns, W = H)")
    axes[0].set_ylabel("time per world (ms, log scale)")
    axes[-1].legend(loc="upper left", fontsize=9)
    fig.suptitle("Time to generate one world, by world size (lower is faster)", fontweight="bold", x=0.5, y=1.02)
    save(fig, "time_vs_size.png")

    # --- speedup vs size ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    panels = [("speedup_vs_1core_total", "GPU total vs. CPU 1 core"),
              ("speedup_vs_allcores_total", "GPU total vs. CPU all cores")]
    for ax, (key, title) in zip(axes, panels):
        for t in present:
            g = by_version(sizes, type=t)["gpu"]
            xs, ys = [num(r, "width") for r in g], [num(r, key) for r in g]
            ax.plot(xs, ys, color=TYPE_COLOR[t], marker=TYPE_MARK[t], label=t, markeredgecolor=SURFACE)
            ax.annotate(f"{ys[-1]:.0f}x", (xs[-1], ys[-1]), textcoords="offset points",
                        xytext=(6, 0), va="center", fontsize=9, color=INK)
        ax.axhline(1, color=MUTED, linewidth=1, linestyle="--")
        ax.text(0.98, 1, "1x = same speed; above this line the GPU is faster ",
                transform=ax.get_yaxis_transform(), va="top", ha="right", fontsize=8.5, color=MUTED)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ticks = sorted({num(r, "width") for r in sizes})
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{int(w)}" for w in ticks], rotation=45)
        ax.minorticks_off()
        ax.set_title(title)
        ax.set_xlabel("world side length (columns, W = H)")
    axes[0].set_ylabel("speedup (times faster, log scale)")
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, 0.92), fontsize=9)
    fig.suptitle("GPU speedup including data copies, by world size", fontweight="bold", y=1.02)
    save(fig, "speedup_vs_size.png")

    # --- GPU time breakdown (share of total) ---
    fig, axes = plt.subplots(1, len(present), figsize=(5.2 * len(present), 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    parts = [("h2d_median_ms", "copy table in", BLUE),
             ("kernel_median_ms", "kernel (the math)", ORANGE),
             ("d2h_median_ms", "copy heights back", AQUA)]
    for ax, t in zip(axes, present):
        g = by_version(sizes, type=t)["gpu"]
        labels = [f"{int(num(r, 'width'))}" for r in g]
        x = np.arange(len(g))
        tot = np.array([sum(num(r, k) for k, _, _ in parts) for r in g])
        bottom = np.zeros(len(g))
        for key, label, color in parts:
            share = 100 * np.array([num(r, key) for r in g]) / tot
            ax.bar(x, share, bottom=bottom, color=color, label=label, width=0.62,
                   edgecolor=SURFACE, linewidth=2)
            for xi, (b, s) in enumerate(zip(bottom, share)):
                if s >= 9:
                    ax.text(xi, b + s / 2, f"{s:.0f}%", ha="center", va="center", fontsize=8.5,
                            color=INK, bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none", alpha=0.85))
            bottom += share
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45)
        ax.set_ylim(0, 100)
        ax.grid(axis="x", visible=False)
        ax.set_title(t)
        ax.set_xlabel("world side length (columns, W = H)")
    axes[0].set_ylabel("share of GPU total time (%)")
    axes[-1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3, fontsize=9)
    fig.suptitle("Where the GPU's time goes: copying vs. computing", fontweight="bold", y=1.02)
    save(fig, "gpu_time_breakdown.png")

    # --- summary tables ---
    summary += ["## RQ1: world size (time in ms, speedup = CPU time / GPU total time)", ""]
    for t in present:
        v = by_version(sizes, type=t)
        cpu1 = {r["width"]: r for r in v["cpu_1core"]}
        cpun = {r["width"]: r for r in v["cpu_all_cores"]}
        summary += [f"**{t}** ({v['cpu_1core'][0]['octaves']} octaves)", "",
                    "| World | CPU 1 core | CPU all cores | GPU kernel | GPU total | vs 1 core | vs all cores | Faster |",
                    "|---|---:|---:|---:|---:|---:|---:|---|"]
        for r in v["gpu"]:
            w = r["width"]
            s1, sn = num(r, "speedup_vs_1core_total"), num(r, "speedup_vs_allcores_total")
            win = "GPU" if min(s1, sn) > 1 else ("CPU" if max(s1, sn) < 1 else "GPU vs 1 core only")
            summary.append(f"| {w} x {r['height']} | {ms(num(cpu1[w], 'median_ms'))} | "
                           f"{ms(num(cpun[w], 'median_ms'))} | {ms(num(r, 'kernel_median_ms'))} | "
                           f"{ms(num(r, 'median_ms'))} | {s1:.1f}x | {sn:.1f}x | {win} |")
        summary.append("")
    summary += ["## RQ4: cost of copying data (GPU, ms)", ""]
    for t in present:
        summary += [f"**{t}**", "",
                    "| World | Bytes copied back | Copy table in | Kernel | Copy heights back | Total | Copying share |",
                    "|---|---:|---:|---:|---:|---:|---:|"]
        for r in by_version(sizes, type=t)["gpu"]:
            summary.append(f"| {r['width']} x {r['height']} | {int(num(r, 'bytes_copied_back')):,} | "
                           f"{ms(num(r, 'h2d_median_ms'))} | {ms(num(r, 'kernel_median_ms'))} | "
                           f"{ms(num(r, 'd2h_median_ms'))} | {ms(num(r, 'median_ms'))} | "
                           f"{num(r, 'transfer_pct'):.1f}% |")
        summary.append("")

# ---------------------------------------------------------------------------
# RQ2: octaves
# ---------------------------------------------------------------------------
octs = load("benchmark_octaves.csv")
if octs:
    v = by_version(octs)
    world = f"{octs[0]['width']} x {octs[0]['height']}"
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2))
    lines = [(v["cpu_1core"], "median_ms"), (v["cpu_all_cores"], "median_ms"),
             (v["gpu"], "median_ms"), (v["gpu"], "kernel_median_ms")]
    for (rows, key), (label, color, marker) in zip(lines, SERIES):
        if rows:
            a.plot([num(r, "octaves") for r in rows], [num(r, key) for r in rows],
                   color=color, marker=marker, label=label, markeredgecolor=SURFACE)
    a.set_yscale("log")
    a.set_title("Time per world")
    a.set_ylabel("time (ms, log scale)")
    a.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    g = v["gpu"]
    xs = [num(r, "octaves") for r in g]
    for key, label, color, marker in [("speedup_vs_1core_total", "vs. CPU 1 core", BLUE, "o"),
                                      ("speedup_vs_allcores_total", "vs. CPU all cores", ORANGE, "s")]:
        ys = [num(r, key) for r in g]
        b.plot(xs, ys, color=color, marker=marker, label=label, markeredgecolor=SURFACE)
        b.annotate(f"{ys[-1]:.0f}x", (xs[-1], ys[-1]), textcoords="offset points", xytext=(6, 0),
                   va="center", fontsize=9, color=INK)
    b.set_ylim(bottom=0)
    b.set_title("GPU total speedup (copies included)")
    b.set_ylabel("speedup (times faster)")
    b.legend(fontsize=9, loc="upper left")
    for ax in (a, b):
        ax.set_xticks(xs)
        ax.set_xlabel("octaves (noise layers per column)")
    fig.suptitle(f"More work per column: hilly terrain, {world}", fontweight="bold", y=1.02)
    save(fig, "octaves.png")

    cpu1 = {r["octaves"]: r for r in v["cpu_1core"]}
    cpun = {r["octaves"]: r for r in v["cpu_all_cores"]}
    summary += [f"## RQ2: octaves (hilly, {world}, ms)", "",
                "| Octaves | CPU 1 core | CPU all cores | GPU kernel | GPU total | vs 1 core | vs all cores | Copying share |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in g:
        o = r["octaves"]
        summary.append(f"| {o} | {ms(num(cpu1[o], 'median_ms'))} | {ms(num(cpun[o], 'median_ms'))} | "
                       f"{ms(num(r, 'kernel_median_ms'))} | {ms(num(r, 'median_ms'))} | "
                       f"{num(r, 'speedup_vs_1core_total'):.1f}x | {num(r, 'speedup_vs_allcores_total'):.1f}x | "
                       f"{num(r, 'transfer_pct'):.1f}% |")
    summary.append("")

# ---------------------------------------------------------------------------
# RQ3: threads per block
# ---------------------------------------------------------------------------
blocks = load("benchmark_blocks.csv")
if blocks:
    g = by_version(blocks)["gpu"]
    worlds = sorted({int(num(r, "width")) for r in g})
    fig, axes = plt.subplots(1, len(worlds), figsize=(4.8 * len(worlds), 4))
    axes = np.atleast_1d(axes)
    for ax, w in zip(axes, worlds):
        rows = [r for r in g if int(num(r, "width")) == w]
        labels = [f"{r['block_x']}x{r['block_y']}\n({r['threads_per_block']})" for r in rows]
        vals = [num(r, "kernel_median_ms") for r in rows]
        bars = ax.bar(np.arange(len(rows)), vals, color=BLUE, width=0.55)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val, ms(val), ha="center", va="bottom",
                    fontsize=9, color=INK)
        ax.set_xticks(np.arange(len(rows)))
        ax.set_xticklabels(labels)
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, max(vals) * 1.15)
        ax.set_title(f"{w} x {w} world")
        ax.set_xlabel("block size (threads per block)")
    axes[0].set_ylabel("GPU kernel time (ms)")
    fig.suptitle("GPU kernel time by threads per block (hilly, lower is faster)", fontweight="bold", y=1.02)
    save(fig, "blocks.png")

    summary += ["## RQ3 (first half): threads per block (hilly, GPU, ms)", "",
                "| World | Block | Threads per block | Kernel | Total | vs 1 core (total) |",
                "|---|---|---:|---:|---:|---:|"]
    for r in g:
        summary.append(f"| {r['width']} x {r['height']} | {r['block_x']} x {r['block_y']} | "
                       f"{r['threads_per_block']} | {ms(num(r, 'kernel_median_ms'))} | "
                       f"{ms(num(r, 'median_ms'))} | {num(r, 'speedup_vs_1core_total'):.1f}x |")
    summary.append("")

# ---------------------------------------------------------------------------
# Correctness and terrain stats
# ---------------------------------------------------------------------------
corr = load("correctness_summary.csv")
if corr:
    cases = sum(int(r["cases"]) for r in corr)
    passed = sum(int(r["passed"]) for r in corr)
    summary += ["## Correctness: CPU vs. GPU on the edge-case sizes", "",
                f"**{passed} of {cases} cases passed.** Each size is run with 2 seeds, 3 origins and "
                "3 terrain types. \"Idle threads\" were launched but had no column, so the bounds "
                "check stopped them.", "",
                "| World | Grid of blocks | Threads launched | Idle threads | Cases | Passed | Largest height difference | Columns that differ |",
                "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in corr:
        summary.append(f"| {r['width']} x {r['height']} | {r['grid_x']} x {r['grid_y']} | "
                       f"{int(r['threads_launched']):,} | {int(r['idle_threads']):,} | {r['cases']} | "
                       f"{r['passed']} | {r['worst_max_diff']} | {r['total_mismatches']} |")
    summary.append("")

stats = load("stats.csv")
if stats:
    summary += ["## Terrain stats (1024 x 1024, main seed)", "",
                "| Type | Octaves | Made by | Matches CPU | Lowest | Highest | Average | Steepest step | Limit | Smooth | Clamped columns |",
                "|---|---:|---|---|---:|---:|---:|---:|---:|---|---:|"]
    for r in stats:
        summary.append(f"| {r['type']} | {r['octaves']} | {r['made_by']} | "
                       f"{'yes' if r['matches_cpu'] == '1' else 'NO'} | {r['hmin']} | {r['hmax']} | "
                       f"{float(r['havg']):.2f} | {r['dmax']} | {r['dmax_limit']} | "
                       f"{'pass' if r['smooth_pass'] == '1' else 'FAIL'} | {r['clamped']} |")
    summary.append("")

# ---------------------------------------------------------------------------
# Heightmap pictures (PNG is much smaller than the BMP and shows on GitHub)
# ---------------------------------------------------------------------------
maps = []
for t in TYPES:
    for path in sorted(RES.glob(f"{t}_*.bin")):
        with open(path, "rb") as f:
            W, H = np.fromfile(f, dtype=np.int32, count=2)
            h = np.fromfile(f, dtype=np.int32).reshape(H, W)
        plt.imsave(OUT / (path.stem + ".png"), h, cmap="gray")
        maps.append((t, h))
        print(f"  saved plots/{path.stem}.png")
        break
if maps:
    fig, axes = plt.subplots(1, len(maps), figsize=(4.6 * len(maps), 4.9))
    axes = np.atleast_1d(axes)
    for ax, (t, h) in zip(axes, maps):
        im = ax.imshow(h, cmap="gray")
        ax.set_title(f"{t}: heights {h.min()} to {h.max()}")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.04, fraction=0.046, label="height (Y)")
    fig.suptitle("Generated heightmaps (dark = low, bright = high)", fontweight="bold", y=1.0)
    save(fig, "heightmaps.png")

(RES / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
print("  saved summary.md")
print(f"Done. Everything is in {RES}")
