"""Plot run.py's results: autokernel's speedup over vLLM TorchTPU's RPA v3, per workload (device time).

uv run --with matplotlib python benchmarks/vs_vllm_rpa/plot.py [benchmarks/results/rpa_vs_vllm_v5e.txt] [out.png]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent.parent
LOG = ROOT / "benchmarks/results/rpa_vs_vllm_v5e.txt"
OUT = ROOT / "benchmarks/results/rpa_vs_vllm_v5e.png"

BAR, INK, MUTED, GRID, SURFACE = "#2a78d6", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
GROUPS = [("decode", "Decode (batch × mean context)"), ("prefill", "Prefill"), ("chunk", "Chunked prefill"),
          ("mixed", "Mixed batches")]  # fmt: skip


def device_times(text: str) -> dict[str, dict[str, float]]:
    """workload -> method -> device seconds, from run.py's output (the last run of each wins)."""
    found: dict[str, dict[str, float]] = {}
    name = None
    for line in text.splitlines():
        if m := re.match(r"^(\S+): \d+ sequences", line):
            name = m.group(1)
        elif (m := re.match(r"\s+(vllm|vllm-prefill|autokernel)\s+wall.*device\s+([\d.]+) us", line)) and name:
            found.setdefault(name, {})[m.group(1)] = float(m.group(2)) * 1e-6
    return {k: v for k, v in found.items() if {"vllm", "autokernel"} <= set(v)}


def label(name: str) -> str:
    if m := re.match(r"decode(\d+)_(\d+)(x?)", name):
        batch, length, fixed = m.groups()
        return f"{batch} × {int(length) // 1024 or 0.5:g}k" + (", all at 8k" if fixed else "")
    if m := re.match(r"prefill(\d+)x(\d+)", name):
        return f"{m.group(1)} × {m.group(2)} tokens"
    if m := re.match(r"chunk(\d+)@(\d+)", name):
        return f"{m.group(1)} tokens after {int(m.group(2)) // 1024}k"
    if m := re.match(r"mixed(\d+)\+(\d+)x(\d+)", name):
        return f"{m.group(1)} decodes + {m.group(2)} × {m.group(3)}"
    return name


def main() -> None:
    log = Path(sys.argv[1]) if len(sys.argv) > 1 else LOG
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else OUT
    times = device_times(log.read_text())
    rows: list[tuple[str, float] | None] = []
    heads: list[tuple[int, str]] = []
    for prefix, title in GROUPS:
        names = [n for n in times if n.startswith(prefix)]
        if names:
            heads.append((len(rows), title))
            rows.append(None)  # (the group's title row)
            rows += [(label(n), times[n]["vllm"] / times[n]["autokernel"]) for n in names]
    fig, ax = plt.subplots(figsize=(8, 0.26 * len(rows) + 1.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    y = list(range(len(rows)))[::-1]
    bars = [(yy, r) for yy, r in zip(y, rows, strict=True) if r is not None]
    speedups = [r[1] for _, r in bars]
    lo, hi = min(min(speedups), 1.0) - 0.08, max(speedups) + 0.1
    # Bars grow from 1x, the break-even: right where autokernel is faster, left where it is slower.
    ax.barh([yy for yy, _ in bars], [s - 1 for s in speedups], left=1.0, height=0.62, color=BAR, zorder=2)
    for (yy, _), s in zip(bars, speedups, strict=True):
        right = s >= 1
        x = s + (0.012 if right else -0.012)
        ax.text(x, yy, f"{s:.2f}×", va="center", ha="left" if right else "right", fontsize=7.5, color=MUTED)
    ax.set_yticks([yy for yy, _ in bars], [r[0] for _, r in bars], fontsize=8, color=INK)
    for i, title in heads:
        ax.text(-0.012, y[i], title, va="center", ha="right", fontsize=9, fontweight="bold", color=INK,
                transform=ax.get_yaxis_transform())  # fmt: skip
    ax.axvline(1.0, color=INK, linewidth=1, zorder=3)
    ax.set_xlim(lo, hi)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.tick_params(axis="both", length=0, colors=MUTED, labelsize=8)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_xlabel("speedup over vLLM's RPA v3 (device time; > 1 is faster)", fontsize=8.5, color=MUTED)
    ax.set_title(
        "Ragged paged attention on a TPU v5e: autokernel vs vLLM TorchTPU\n"
        "Llama 3 8B attention (32 q / 8 kv heads of 128), bf16, pages of 128; RPA v3 as vLLM's runner calls it",
        fontsize=10,
        color=INK,
        loc="right",
    )
    fig.tight_layout()
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
