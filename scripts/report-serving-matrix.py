#!/usr/bin/env python3
"""Verify the paired serving runs and render their public figures and tables."""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median

MODELS = ("Inkling", "Qwen")
PREFIX_64K = "prefix-64k-cold-warm"
FIELDS = {
    "wall_s": "batch_wall_s",
    "prompt_tokens": "requested_prompt_tokens",
    "cached_tokens": "prompt_tokens_cached",
    "computed_tokens": "prefill_tokens_computed",
    "output_tokens": "completion_tokens",
    "prefill_s": "native_prefill_seconds",
    "decode_s": "native_decode_seconds",
    "prefill_tps": "native_prefill_compute_tokens_per_s",
    "decode_tps": "native_weighted_decode_tokens_per_s",
    "wave_prefill_tps": "aggregate_prefill_wave_tokens_per_s",
    "wave_output_tps": "legacy_aggregate_output_tokens_per_s",
    "overlap_decode_tps": "fully_overlapped_aggregate_decode_tokens_per_s",
    "energy_j": "card_energy_j",
    "queue_s": "queue_time_sum_s",
    "peak_running": "peak_running",
    "peak_waiting": "peak_waiting",
    "preemptions": "preemptions",
    "draft_tokens": "speculative_draft_tokens",
    "accepted_tokens": "speculative_accepted_tokens",
}


def load_json(path):
    return json.loads(path.read_bytes())


def require(check, message):
    if not check:
        raise ValueError(message)


def read_wave(folder):
    wave = load_json(folder / "summary.json")
    case = wave["scenario"]
    requests = wave["requests"]
    n = case["concurrency"]
    require(
        wave["finished_requests_delta"] == len(requests) == n
        and wave["completion_tokens"] == n * 1024
        and case["output_tokens"] == 1024
        and wave["requested_prompt_tokens"] == wave["logical_prompt_tokens_observed"]
        and wave["preemptions"] == 0
        and wave["admission_expectation_met"]
        and all(
            wave[k]
            for k in (
                "all_outputs_nonempty",
                "all_prompt_counts_match",
                "all_completion_counts_exact",
                "all_finish_reasons_length",
            )
        ),
        f"Failed request accounting in {folder}",
    )
    result = {
        "scenario": case["name"],
        "group": case["group"],
        "nominal_tokens": case["prompt_tokens"],
        "clients": n,
        "repeat": wave["repeat"],
    }
    result.update({short: wave[key] for short, key in FIELDS.items()})
    result["client_ttft_s"] = [r["ttft_s"] for r in requests]
    result["client_wall_s"] = [r["wall_s"] for r in requests]
    result["stream_events"] = [r["events"] for r in requests]
    result["ttft_s"] = median(result["client_ttft_s"])
    result["client_prompt_tokens"] = [r["usage"]["prompt_tokens"] for r in requests]
    result["source_sha256"] = []
    settings = []
    for i, request in enumerate(requests, 1):
        require(
            request["finish_reason"] == "length"
            and request["content_chars"] > 0
            and request["usage"]["completion_tokens"] == 1024
            and not request["reasoning_chars"]
            and request["usage"]["completion_tokens_details"]["reasoning_tokens"] == 0,
            f"Incomplete output or unexpected thinking in {folder} request {i}",
        )
        metadata = load_json(folder / f"request-{i}.json")
        digest = hashlib.sha256((folder / f"prompt-{i}.txt").read_bytes()).hexdigest()
        require(metadata["prompt_sha256"] == digest, f"Prompt digest mismatch in {folder}")
        result["source_sha256"].append(digest)
        translated = {
            "model",
            "messages",
            "reasoning_effort",
            "thinking_token_budget",
            "chat_template_kwargs",
        }
        settings.append({k: v for k, v in metadata.items() if k not in translated})
    result["workload_sha256"] = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    return result


def matrix(inkling, reference, isolated):
    manifest = load_json(inkling / "manifest.json")
    end = load_json(inkling / "completed.json")["end"]
    runtime = load_json(inkling / "runtime.json")
    receipt = load_json(isolated / "manifest.json")
    require(
        load_json(isolated / "completed.json")["end"] > end
        and receipt["container"]["id"] != manifest["container"]["id"]
        and load_json(isolated / "runtime.json") == runtime,
        "Isolated run needs the same recipe on a fresh worker",
    )
    data = {
        "runtime": runtime,
        "start": manifest["start"],
        "end": end,
        "reference": manifest["reference_revision"],
        "fixtures": manifest["fixture_inventories"],
        "power_cap_w": manifest["power_w"],
        "measured": {m: [] for m in MODELS},
    }
    qref = load_json(reference / "manifest.json")
    data["qwen_release"] = {
        "start": qref["started_at"],
        "container": qref["container"],
        "sampling": qref["sampling"],
        "chat_template_kwargs": qref["chat_template_kwargs"],
    }
    qfolders = sorted(reference.glob("*/summary.json"))
    folders = sorted((inkling / "matrix").glob("*/summary.json"))
    require(
        len(folders) == 70 and [p.parent.name for p in folders] == [p.parent.name for p in qfolders],
        "Incomplete or mismatched matrix",
    )
    for a, b in zip(folders, qfolders):
        ink, qwen = read_wave(a.parent), read_wave(b.parent)
        require(
            ink["workload_sha256"] == qwen["workload_sha256"] and ink["clients"] == qwen["clients"],
            f"Unpaired workload {a.parent.name}",
        )
        for model, wave in zip(MODELS, (ink, qwen)):
            data["measured"][model].append(wave)
    warmups = sorted((inkling / "warmups").glob("*/summary.json"))
    require(len(warmups) == 17, "Missing warmups")
    warm = [read_wave(path.parent) for path in warmups]
    require(sum(w["clients"] for w in warm) == 35, "Incomplete warmup requests")
    for row, path in zip(warm, warmups):
        src = reference / "warmups" / path.parent.name.replace("-r1", "-r0")
        require(row["workload_sha256"] == read_wave(src)["workload_sha256"], f"Unpaired warmup {src.name}")
    data["warmup_waves"] = len(warm)
    data["warmup_requests"] = sum(w["clients"] for w in warm)
    data["isolated_64k"] = [read_wave(isolated / "matrix" / f"{PREFIX_64K}-r{i}") for i in (1, 2, 3)]
    phases = data["isolated_64k"]
    require(
        phases[0]["cached_tokens"] == 0
        and all(p["source_sha256"] == phases[0]["source_sha256"] for p in phases[1:]),
        "Isolated control needs an empty cache and byte-identical resends",
    )
    data["isolated_64k_both_resends_hit"] = all(p["cached_tokens"] > 0 for p in phases[1:])
    for model, waves in data["measured"].items():
        require(
            len({w["scenario"] for w in waves}) == 20 and sum(w["clients"] for w in waves) == 124,
            f"Incomplete {model} scenarios",
        )
    return data


def select(waves, name):
    return sorted((w for w in waves if w["scenario"] == name), key=lambda w: w["repeat"])


def mid(rows, field):
    return median(w[field] for w in rows)


def context(waves):
    return sorted(
        {(w["nominal_tokens"], w["scenario"]) for w in waves if w["group"] in {"phase-sweep", "full-context"}}
    )


def concurrent_point(rows, field):
    if field == "mean_queue_s":
        return median(r["queue_s"] / r["clients"] for r in rows)
    if field == "overlap_decode_tps" and rows[0]["clients"] == 1:
        return mid(rows, "decode_tps")
    rates = [r[field] for r in rows if r[field] is not None]
    return median(rates) if rates else float("nan")


def draw(data, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    plt.rcParams.update(
        {
            "font.size": 11,
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.spines.bottom": False,
            "axes.labelcolor": "#394454",
            "ytick.color": "#394454",
            "xtick.color": "#394454",
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.axisbelow": True,
            "figure.facecolor": "#f8fafc",
            "axes.facecolor": "#f8fafc",
        }
    )
    colors = {"Inkling": "#006e70", "Qwen": "#a44822"}
    styles = {"Inkling": "-", "Qwen": "--"}
    markers = {"Inkling": "o", "Qwen": "s"}
    models = data["measured"]
    legend = [
        Line2D(
            [],
            [],
            color=colors[m],
            linestyle=styles[m],
            marker=markers[m],
            label=("Inkling 2 Mini · DSpark" if m == "Inkling" else "Qwen3.8-27B · MTP"),
        )
        for m in MODELS
    ]

    def base(ax, title, unit):
        ax.set_title(title, fontsize=13, pad=11)
        ax.set_ylabel(unit, fontsize=10)
        ax.grid(axis="y", color="#d9dfe5", lw=0.8)
        ax.tick_params(length=0, pad=5)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
        ax.set_ylim(bottom=0)

    def banner(fig, title, subtitle):
        fig.text(0.07, 0.95, title, fontsize=21, weight="bold", color="#14283b", va="top")
        fig.text(0.07, 0.885, subtitle, fontsize=11, color="#495564", va="top")
        fig.legend(
            handles=legend,
            loc="upper left",
            bbox_to_anchor=(0.067, 0.842),
            frameon=False,
            ncols=2,
            borderaxespad=0,
        )

    fig, axs = plt.subplots(2, 2, figsize=(12, 8.5))
    fig.subplots_adjust(top=0.72, left=0.085, right=0.965, bottom=0.13, hspace=0.67, wspace=0.29)
    banner(
        fig,
        "Single-request serving across context lengths",
        "Intel Arc Pro B70 · 180 W · Same source text · 1,024 sampled output tokens per request",
    )
    labels = [str(n // 1024) + "K" if n > 512 else "512" for n, _ in context(models["Inkling"])]
    for ax, (field, title, unit) in zip(
        axs.flat[:3],
        (
            ("prefill_tps", "Prompt processing", "Computed tokens/s"),
            ("decode_tps", "Decode after the first token", "Output tokens/s"),
            ("ttft_s", "Time to first token", "Seconds"),
        ),
    ):
        for m in MODELS:
            sets = [select(models[m], name) for _, name in context(models[m])]
            vals = [mid(rows, field) for rows in sets]
            ax.vlines(
                range(len(vals)),
                [min(w[field] for w in rows) for rows in sets],
                [max(w[field] for w in rows) for rows in sets],
                lw=1.3,
                alpha=0.4,
                color=colors[m],
            )
            ax.plot(range(len(vals)), vals, styles[m], marker=markers[m], color=colors[m], lw=2, ms=4)
        ax.set_xticks(range(len(labels)), labels, fontsize=9)
        ax.set_xlabel("Qwen input budget", fontsize=10)
        base(ax, title, unit)
    ax = axs.flat[3]
    for m, offset in zip(MODELS, (-0.19, 0.19)):
        vals = [mid(select(models[m], f"prefix-{k}k-cold-warm")[1:], "ttft_s") for k in (16, 64)]
        bars = ax.bar([x + offset for x in (0, 1)], vals, width=0.34, color=colors[m])
        ax.bar_label(bars, fmt="%.2f s", padding=3, fontsize=9)
    ax.margins(y=0.27)
    ax.set_xticks([0, 1], ["16K prefix", "64K prefix"])
    base(ax, "First visible output on prefix resends", "Seconds")
    fig.text(
        0.07,
        0.034,
        "Points show medians and whiskers span the observed waves. Prefix bars use the two resends.",
        fontsize=10,
    )
    fig.savefig(out / "serving-context.png", dpi=150)
    plt.close(fig)

    fig, axs = plt.subplots(3, 2, figsize=(12, 12.8))
    fig.subplots_adjust(top=0.765, left=0.085, right=0.94, bottom=0.10, hspace=0.76, wspace=0.31)
    banner(
        fig,
        "Serving one to four simultaneous requests",
        "Intel Arc Pro B70 · 180 W · Inkling admits one sequence at a time · Qwen supports batched decode",
    )
    # Group by context, with separate model curves and one label per concurrency point.
    fields = (
        ("wave_prefill_tps", "Input through the last first token", "Prompt tokens/s"),
        ("overlap_decode_tps", "Combined decode during full overlap", "Output tokens/s"),
        ("ttft_s", "Typical client time to first token", "Seconds"),
        ("decode_tps", "Decode during active request time", "Output tokens/s per request"),
        ("wave_output_tps", "Delivered output over the whole wave", "Output tokens/s"),
        ("mean_queue_s", "Scheduler queue time per request", "Seconds"),
    )
    for ax, (field, title, unit) in zip(axs.flat, fields):
        for j, k in enumerate((2, 4, 16)):
            for m in MODELS:
                sets = [select(models[m], f"phase-{k}k-c1")] + [
                    select(models[m], f"concurrency-{k}k-c{c}") for c in (2, 3, 4)
                ]
                ax.plot(
                    [j * 5 + i for i in range(4)],
                    [concurrent_point(rows, field) for rows in sets],
                    styles[m],
                    marker=markers[m],
                    color=colors[m],
                    lw=1.9,
                    ms=4,
                )
            ax.text(
                (j * 5 + 1.5),
                -0.19,
                f"{k}K prompt",
                transform=ax.get_xaxis_transform(),
                ha="center",
                fontsize=9,
                weight="bold",
            )
        ax.set_xticks([j * 5 + i for j in range(3) for i in range(4)], ["1", "2", "3", "4"] * 3, fontsize=9)
        base(ax, title, unit)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
    fig.text(
        0.07,
        0.028,
        "Groups show 1–4 clients. Inkling has no full decode overlap at 2–4 clients. At C1 the overlap panel uses native decode.",
        fontsize=10,
    )
    fig.savefig(out / "serving-concurrency.png", dpi=150)
    plt.close(fig)


def total(waves):
    wall = sum(r["wall_s"] for r in waves)
    count = sum(r["output_tokens"] for r in waves)
    return {
        "wall": wall,
        "outputs": count,
        "rate": count / wall,
        "prompts": sum(r["prompt_tokens"] for r in waves),
        "energy": sum(r["energy_j"] for r in waves),
        "running": max(r["peak_running"] for r in waves),
        "waiting": max(r["peak_waiting"] for r in waves),
    }


def fmt(value, decimals=2):
    return "—" if value is None else f"{value:,.{decimals}f}"


def pair(rows, field, decimals):
    return " / ".join(fmt(mid(model_waves, field), decimals) for model_waves in rows)


def tables(data):
    models = data["measured"]
    lines = [
        "## Full-matrix totals",
        "",
        "| Release | Input tokens | Output tokens | Summed wave time | Delivered output | Card energy | Peak running / waiting |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for m in MODELS:
        t = total(models[m])
        lines.append(
            f"| {m} | {t['prompts']:,} | {t['outputs']:,} | {t['wall'] / 60:.2f} min | "
            f"{t['rate']:.2f} tok/s | {t['energy'] / 3600000:.3f} kWh | "
            f"{t['running']:.0f} / {t['waiting']:.0f} |"
        )
    lines.extend(
        [
            "",
            "## One client",
            "",
            "Each row shows the median over the indicated waves on identical input text. The last row is one observation. K denotes 1,024 tokens.",
            "",
            "| Qwen budget | Waves | Actual input, Inkling / Qwen | Prefill tok/s, Inkling / Qwen | Decode tok/s, Inkling / Qwen | First token (s), Inkling / Qwen |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for n, name in context(models["Inkling"]):
        rows = [select(models[m], name) for m in MODELS]
        label = str(n // 1024) + "K" if n >= 1024 else str(n)
        lines.append(
            f"| {label} | {len(rows[0])} | {pair(rows, 'prompt_tokens', 0)} | {pair(rows, 'prefill_tps', 0)} | "
            f"{pair(rows, 'decode_tps', 2)} | {pair(rows, 'ttft_s', 2)} |"
        )
    lines.extend(
        [
            "",
            "## Simultaneous clients",
            "",
            "Wave throughput includes queueing, prompt processing and all output. First-token latency is the median of per-wave client medians. The last columns measure aggregate decode while all clients are generating. A dash means there was no measurable overlap interval.",
            "",
            "| Budget | Clients | Wave output tok/s, Inkling / Qwen | First token (s), Inkling / Qwen | Wave time (s), Inkling / Qwen | Fully overlapping decode tok/s, Inkling / Qwen |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for k in (2, 4, 16):
        for c in (2, 3, 4):
            rows = [select(models[m], f"concurrency-{k}k-c{c}") for m in MODELS]
            overlaps = []
            for rr in rows:
                rates = [r["overlap_decode_tps"] for r in rr if r["overlap_decode_tps"] is not None]
                overlaps.append(fmt(median(rates) if rates else None))
            lines.append(
                f"| {k}K | {c} | {pair(rows, 'wave_output_tps', 2)} | {pair(rows, 'ttft_s', 2)} | "
                f"{pair(rows, 'wall_s', 2)} | {' / '.join(overlaps)} |"
            )
    lines.extend(
        [
            "",
            "## Prefix reuse",
            "",
            "The first request seeds each prefix and the next two resend the same text. Earlier matrix scenarios can leave a partial prefix resident. The additional Inkling worker starts with an empty cache.",
            "",
            "| Release | Nominal input | Actual input | Cached tokens, first / second / third | Time to first token (s), first / second / third |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for m, label, rows in [
        (m, f"{k}K matrix", select(models[m], f"prefix-{k}k-cold-warm")) for m in MODELS for k in (16, 64)
    ] + [("Inkling", "64K isolated", data["isolated_64k"])]:
        hits = " / ".join(fmt(r["cached_tokens"], 0) for r in rows)
        ttfts = " / ".join(fmt(r["ttft_s"], 2) for r in rows)
        lines.append(f"| {m} | {label} | {rows[0]['prompt_tokens']:,} | {hits} | {ttfts} |")
    lines.extend(
        [
            "",
            "| Release | Prefix | Native prefill tok/s, first → resends | Native decode tok/s, first → resends |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for m in MODELS:
        for k in (16, 64):
            rows = select(models[m], f"prefix-{k}k-cold-warm")
            rates = [f"{fmt(rows[0][f])} → {fmt(mid(rows[1:], f))}" for f in ("prefill_tps", "decode_tps")]
            lines.append(f"| {m} | {k}K | {' | '.join(rates)} |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inkling-run", type=Path)
    parser.add_argument("--reference-run", type=Path)
    parser.add_argument("--isolated-run", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    src = args.output / "waves.json"
    raw = [args.inkling_run, args.reference_run, args.isolated_run]
    if any(raw):
        if not all(raw):
            parser.error("Supply all three raw run directories or render the published waves.json")
        data = matrix(*raw)
        src.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    else:
        data = load_json(src)
    draw(data, args.output)
    (args.output / "measurements.md").write_text(tables(data))


if __name__ == "__main__":
    main()
