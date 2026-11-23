#!/usr/bin/env python3
"""Verify paired 4K concurrency waves and plot pure decode throughput."""

import argparse
import json
from pathlib import Path
from runpy import run_path
from statistics import median

from serving_decode import active_decode

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ["phase-4k-c1", *(f"concurrency-4k-c{n}" for n in (2, 3, 4))]
ARCHIVE = ROOT / "benchmarks/2026-11-23/serving/waves.json"


def extract(ordinary, dspark, qwen, two_slot=None):
    read_wave = run_path(str(ROOT / "scripts/report-serving-matrix.py"))["read_wave"]
    archived = json.loads(ARCHIVE.read_text())["measured"]["Inkling"]
    references = [w for w in archived if w["scenario"] in SCENARIOS]
    if len(references) != 14:
        raise ValueError("Expected five C1 waves and three at C2, C3 and C4")
    profiles, receipts = {}, {}
    folders = {"Inkling ordinary": ordinary, "Inkling DSpark B7": dspark, "Qwen R12": qwen}
    if two_slot is not None:
        folders["Observatory DSpark B7"] = two_slot
    for label, folder in folders.items():
        waves = []
        if label != "Qwen R12":
            run = json.loads((folder / "manifest.json").read_text())
            receipt = json.loads((folder / "runtime.json").read_text())
            receipts[label] = dict(start=run["start"], end=json.loads((folder / "completed.json").read_text())["end"],
                                   power_w=run["power_w"], image=receipt["image_id"], files=receipt["file_sha256"],
                                   checkout_head_at_start=receipt["recipe_revision"],
                                   max_num_seqs=run["container"]["max_num_seqs"],
                                   num_speculative_tokens=run["container"]["num_speculative_tokens"])
            if receipts[label]["max_num_seqs"] != (2 if label == "Observatory DSpark B7" else 4):
                raise ValueError(f"Unexpected worker capacity in {folder}")
        for ref in references:
            if label == "Observatory DSpark B7" and ref["scenario"] not in SCENARIOS[:2]:
                continue
            name = f"{ref['scenario']}-r{ref['repeat']}"
            wave_dir = folder / name if label == "Qwen R12" else folder / "matrix" / name
            source = read_wave(wave_dir)
            if source["workload_sha256"] != ref["workload_sha256"] or source["source_sha256"] != ref["source_sha256"]:
                raise ValueError(f"Unmatched prompts or settings in {wave_dir}")
            summary = json.loads((wave_dir / "summary.json").read_text())
            result = active_decode(wave_dir, summary)
            if source["peak_running"] != result["clients"] or not result["aggregate_decode_tps"]:
                raise ValueError(f"No interval with every client decoding in {wave_dir}")
            waves.append(dict(scenario=ref["scenario"], repeat=ref["repeat"],
                              workload_sha256=ref["workload_sha256"], **result))
        profiles[label] = waves
    return dict(profiles=profiles, runs=receipts)


def draw(data, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.1))
    fig.subplots_adjust(left=.07, right=.975, top=.72, bottom=.26, wspace=.19)
    fig.text(.065, .90, "Inkling 2 Mini  |  Decode throughput", fontsize=22, weight="bold", color="#172c3c")
    fig.text(.065, .83, "Intel Arc Pro B70  ·  180 W  ·  4K code review  ·  1,024 sampled output tokens per request",
             fontsize=11, color="#435767")
    styles = [("Inkling DSpark B7", "Adaptive DSpark B7 (4 slots)", "#00766f", "o"),
              ("Inkling ordinary", "Ordinary decode (4 slots)", "#5a60a6", "D")]
    for ax, field, title in zip(axes, ("aggregate_decode_tps", "mean_decode_tps_per_client"),
                                ("Combined output", "Mean per request")):
        for key, legend, color, marker in styles:
            waves = data["profiles"][key]
            rates = [median(w[field] for w in waves if w["scenario"] == name) for name in SCENARIOS]
            ax.plot(range(1, 5), rates, color=color, marker=marker, lw=2.4, ms=6.5, label=legend)
        if "Observatory DSpark B7" in data["profiles"]:
            waves = data["profiles"]["Observatory DSpark B7"]
            rates = [median(w[field] for w in waves if w["clients"] == n) for n in (1, 2)]
            ax.scatter([1, 2], rates, s=138, facecolors="none", edgecolors="#263a45",
                       linewidths=1.4, zorder=4, label="Observatory retest (2 slots, DSpark)")
        ax.set_title(title + " · tok/s", loc="left", weight="bold", fontsize=12, color="#172c3c", pad=8)
        ax.set_xticks(range(1, 5))
        ax.set_xlim(.76, 4.24)
        ax.set_xlabel("Simultaneous requests", color="#435767")
        ax.set_ylim(0, 135 if field == "aggregate_decode_tps" else 94)
        ax.yaxis.set_major_locator(MultipleLocator(30 if field == "aggregate_decode_tps" else 20))
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#bac8cf")
        ax.tick_params(axis="both", length=0, pad=6, colors="#41535f")
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#dce4e8")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center",
               bbox_to_anchor=(.5, .09), ncol=3, frameon=False)
    fig.text(.065, .045, "Median across waves. Only intervals with all requests decoding count toward throughput.",
             fontsize=9, color="#435767")
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ordinary", type=Path)
    p.add_argument("--dspark", type=Path)
    p.add_argument("--reference-run", type=Path)
    p.add_argument("--two-slot", type=Path, help="Optional DSpark production retest on a two-slot worker.")
    p.add_argument("--out", type=Path, default=ROOT / "benchmarks/2026-11-23/batching")
    args = p.parse_args()
    raw = (args.ordinary, args.dspark, args.reference_run)
    if any(raw) and not all(raw):
        p.error("Provide all three raw run directories or none")
    if args.two_slot and not all(raw):
        p.error("--two-slot requires the raw run directories")
    args.out.mkdir(exist_ok=True, parents=True)
    dest = args.out / "waves.json"
    if all(raw):
        dest.write_text(json.dumps(extract(*raw, args.two_slot), indent=2) + "\n")
    draw(json.loads(dest.read_text()), args.out / "serving-4k.png")
