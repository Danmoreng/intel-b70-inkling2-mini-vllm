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


def extract(ordinary, dspark, qwen):
    read_wave = run_path(str(ROOT / "scripts/report-serving-matrix.py"))["read_wave"]
    archived = json.loads(ARCHIVE.read_text())["measured"]["Inkling"]
    references = [w for w in archived if w["scenario"] in SCENARIOS]
    if len(references) != 14:
        raise ValueError("Expected five C1 waves and three at C2, C3 and C4")
    profiles, receipts = {}, {}
    for label, folder in (("Inkling ordinary", ordinary), ("Inkling DSpark B7", dspark), ("Qwen R12", qwen)):
        waves = []
        if label != "Qwen R12":
            run = json.loads((folder / "manifest.json").read_text())
            receipt = json.loads((folder / "runtime.json").read_text())
            receipts[label] = dict(start=run["start"], end=json.loads((folder / "completed.json").read_text())["end"],
                                   power_w=run["power_w"], image=receipt["image_id"], files=receipt["file_sha256"],
                                   checkout_head_at_start=receipt["recipe_revision"],
                                   max_num_seqs=run["container"]["max_num_seqs"],
                                   num_speculative_tokens=run["container"]["num_speculative_tokens"])
        for ref in references:
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
    from matplotlib.ticker import MaxNLocator

    fig, axes = plt.subplots(1, 2, figsize=(10.3, 4.8))
    fig.subplots_adjust(left=.078, right=.977, top=.75, bottom=.25, wspace=.29)
    fig.text(.065, .922, "Pure decode throughput on one Intel Arc Pro B70", fontsize=19, weight="bold")
    fig.text(.065, .855, "4K code review  /  1,024 sampled output tokens per client  /  180 W", fontsize=11)
    styles = [("Inkling DSpark B7", "#006f73", "o", "-"), ("Inkling ordinary", "#5668b0", "D", "-"),
              ("Qwen R12", "#a94e18", "^", ":")]
    for ax, field, title in zip(axes, ("aggregate_decode_tps", "mean_decode_tps_per_client"),
                                ("All active clients · tokens/s", "Mean per active client · tokens/s")):
        for label, color, marker, line in styles:
            waves = data["profiles"][label]
            rates = [median(w[field] for w in waves if w["scenario"] == name) for name in SCENARIOS]
            ax.plot(range(1, 5), rates, color=color, marker=marker, linestyle=line, lw=2, label=label)
        ax.set_title(title, loc="left", weight="bold", fontsize=12)
        ax.set_xticks(range(1, 5))
        ax.set_xlabel("Simultaneous clients")
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.2)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center",
               bbox_to_anchor=(.5, .075), ncol=3, frameon=False)
    fig.text(.065, .04, "Median measured wave. Counted only while every client was generating output.", fontsize=9)
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ordinary", type=Path)
    p.add_argument("--dspark", type=Path)
    p.add_argument("--reference-run", type=Path)
    p.add_argument("--out", type=Path, default=ROOT / "benchmarks/2026-11-23/batching")
    args = p.parse_args()
    raw = (args.ordinary, args.dspark, args.reference_run)
    if any(raw) and not all(raw):
        p.error("Provide all three raw run directories or none")
    args.out.mkdir(exist_ok=True, parents=True)
    dest = args.out / "waves.json"
    if all(raw):
        dest.write_text(json.dumps(extract(*raw), indent=2) + "\n")
    draw(json.loads(dest.read_text()), args.out / "serving-4k.png")
