"""Render the checked-in measurements. No fitted or invented learning points."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "artifacts" / "data"
OUT = ROOT / "artifacts" / "charts"
COLORS = ["#a58d7a", "#728560", "#d3a72f"]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def wilson(successes, n):
    z = 1.96
    p = successes / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return center - half, center + half


def setup():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "figure.facecolor": "#f5f1e7",
            "axes.facecolor": "#f5f1e7",
            "text.color": "#233c32",
            "axes.labelcolor": "#233c32",
            "xtick.color": "#637063",
            "ytick.color": "#637063",
            "axes.edgecolor": "#cccdbf",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
        }
    )
    OUT.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=170)
    fig.savefig(OUT / f"{name}.svg")
    plt.close(fig)


def main():
    setup()
    experiments = read(DATA / "experiments.json")
    banana = [e for e in experiments if e["stage"] == "banana"]
    fig, ax = plt.subplots(figsize=(12, 6.8))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.77, bottom=0.2)
    fig.text(0.09, 0.925, "peel.  /  THE LEARNING RECORD", fontsize=11, weight="bold")
    fig.text(
        0.09, 0.853, "Small model. Measurable progress.", fontsize=24, weight="bold"
    )
    for e, color in zip(banana, COLORS):
        curve = e["curve"]
        ax.plot(
            [p["global_step"] / 1000 for p in curve],
            [p["eval_success_rate"] * 100 for p in curve],
            marker="o",
            ms=4,
            lw=2.2,
            color=color,
            label=e["label"],
        )
    ax.set(
        xlabel="PPO environment interactions (thousands)",
        ylabel="Validation extraction success (%)",
        ylim=(-3, 103),
    )
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(axis="y", color="#ddded2", linestyle="--", zorder=0)
    ax.legend(
        loc="lower right",
        frameon=True,
        facecolor="#f5f1e7",
        edgecolor="#ddded2",
        fontsize=10,
    )
    fig.text(
        0.09,
        0.075,
        "32 fixed validation rooms · seed 11 · 77,606 parameters · RTX 3070 Ti Laptop",
        fontsize=10,
    )
    fig.text(
        0.09,
        0.037,
        "Warm-up appears at 0 PPO steps and adds data + compute. Lines join measured points; one run per recipe.",
        fontsize=9,
        color="#637063",
    )
    save(fig, "learning-curves")

    results = read(DATA / "test-results.json")
    champions = [r for r in results if r["phase"] == "champion"]
    fig, ax = plt.subplots(figsize=(12, 6.8))
    fig.subplots_adjust(left=0.1, right=0.96, top=0.77, bottom=0.25)
    fig.text(
        0.1, 0.925, "peel.  /  ROOMS JOYCE HASN'T SEEN", fontsize=11, weight="bold"
    )
    fig.text(0.1, 0.853, "The test comes after the choice.", fontsize=24, weight="bold")
    for i, r in enumerate(champions):
        s = r["summary"]
        lo, hi = wilson(s["successes"], s["episodes"])
        p = s["success_rate"] * 100
        ax.bar(i, p, width=0.55, color=(COLORS + ["#274b42"])[i], zorder=3)
        ax.errorbar(
            i,
            p,
            yerr=[[p - lo * 100], [hi * 100 - p]],
            color="#233c32",
            capsize=5,
            zorder=4,
        )
        ax.text(
            i,
            min(107, hi * 100 + 4),
            f"{p:.1f}%\n{s['successes']}/{s['episodes']}",
            ha="center",
            fontsize=11,
        )
    labels = [
        next(e["label"] for e in experiments if e["id"] == r["run"]) for r in champions
    ]
    labels = [
        s.replace(" · ", "\n")
        .replace("Demonstrations + PPO", "Demonstrations\n+ PPO")
        .replace("Camera-stage exploration", "Camera-stage\nexploration")
        for s in labels
    ]
    ax.set_xticks(range(len(labels)), labels, fontsize=10)
    ax.set(ylabel="Test extraction success (%)", ylim=(0, 115))
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(axis="y", color="#ddded2", linestyle="--", zorder=0)
    fig.text(
        0.1,
        0.105,
        "Validation-selected champions · 128 test rooms per model · bars show 95% Wilson intervals",
        fontsize=10,
    )
    fig.text(
        0.1,
        0.055,
        "Camera is a different, harder task and inherits banana weights. Test results were not used to select checkpoints.",
        fontsize=9,
        color="#637063",
    )
    save(fig, "test-results")

    memories = [read(DATA / f"memory-seed{s}.json") for s in (123, 124, 125)]
    keys = [
        "transformer_history_accuracy",
        "history_removed_accuracy",
        "memoryless_last_observation_accuracy",
    ]
    fig, ax = plt.subplots(figsize=(10, 6.2))
    fig.subplots_adjust(left=0.1, right=0.96, top=0.75, bottom=0.24)
    fig.text(
        0.1, 0.915, "peel.  /  A SEPARATE MEMORY CHECK", fontsize=11, weight="bold"
    )
    fig.text(
        0.1, 0.835, "Same view. Different earlier cue.", fontsize=23, weight="bold"
    )
    for i, key in enumerate(keys):
        values = [m[key] * 100 for m in memories]
        ax.bar(
            i,
            np.mean(values),
            width=0.53,
            color=["#d3a72f", "#a58d7a", "#728560"][i],
            zorder=3,
        )
        ax.scatter(
            np.array([-0.07, 0, 0.07]) + i, values, color="#233c32", s=28, zorder=4
        )
        ax.text(
            i, max(values) + 5, f"{np.mean(values):.1f}% mean", ha="center", fontsize=11
        )
    ax.axhline(50, color="#637063", ls="--", lw=1)
    ax.set_xticks(
        [0, 1, 2],
        [
            "Temporal transformer",
            "Same weights,\nhistory removed",
            "Separately trained\nmemoryless policy",
        ],
        fontsize=10,
    )
    ax.set(ylabel="Held-out cue accuracy (%)", ylim=(0, 115))
    ax.set_yticks([0, 25, 50, 75, 100])
    fig.text(
        0.1,
        0.10,
        "3 seeds · 1,600 training + 400 test examples per seed · dashed line: 50% chance",
        fontsize=10,
    )
    fig.text(
        0.1,
        0.05,
        "Supervised diagnostic, not heist RL. This does not establish that the heist policy relies on memory.",
        fontsize=9,
        color="#637063",
    )
    save(fig, "memory-probe")

    lines = [
        "# Experiment record",
        "",
        "Measured locally on September 5, 2026. These are exploratory results, not replicated algorithm benchmarks.",
        "",
        "## Training recipes and selection",
        "",
        "| Run | Stage | Demonstrations / epochs | PPO steps | Best validation | Selected step | PPO log duration |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for e in experiments:
        c, selected = e["config"], e["champion"]
        records = [
            json.loads(line)
            for line in (DATA / "runs" / e["id"] / "metrics.jsonl")
            .read_text()
            .splitlines()
        ]
        last = [r for r in records if r["phase"] == "ppo"][-1]
        score = selected["validation"]["eval_successes"] if selected else 0
        lines.append(
            f"| {e['label']} | {e['stage']} | {c['bc_episodes']:,} / {c['bc_epochs']} | {c['total_steps']:,} | {score}/32 ({score / 32:.1%}) | {selected['global_step'] if selected else 'none':,} | {last['seconds']:.1f} s |"
        )
    lines += [
        "",
        "PPO duration is the last update's wall-clock field: it excludes demonstration collection/training, initial evaluation, and the final evaluation/export. It includes intermediate evaluations and checkpoint writes. It is not total time-to-solution. The camera run also has `runtime.json` with end-to-end elapsed time and peak CUDA allocation; that run shared CPU resources with final banana evaluation.",
        "",
        "The first two banana recipes use 131,072 PPO interactions and the same seed/validation rooms. The refined recipe doubles that budget, doubles demonstrations to 2,048, uses 20 imitation epochs, and reduces the PPO learning rate from 3e-4 to 5e-5. Therefore the improvement cannot be attributed to just one of those changes. The camera run uses weights from the refined banana champion, 2,048 new camera demonstrations, 16 imitation epochs, and 131,072 PPO interactions. Its score is not directly comparable to the easier banana task.",
        "",
        "## Untouched test rooms",
        "",
        "Checkpoints were selected on validation before test evaluation. All test models use seeds 2,000,000–2,000,127. Re-evaluation exports reuse cached results only when checkpoint SHA256 matches. Every episode's seed, outcome, return, and length is retained in `artifacts/data/runs/*/test-*.json`.",
        "",
        "| Run | Weights | Escapes / 128 | Success | 95% Wilson interval |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in results:
        s = r["summary"]
        lo, hi = wilson(s["successes"], s["episodes"])
        lines.append(
            f"| {r['run']} | {r['phase']} | {s['successes']}/{s['episodes']} | {s['success_rate']:.1%} | {lo:.1%}–{hi:.1%} |"
        )
    lines += [
        "",
        "![Untouched test results](../artifacts/charts/test-results.png)",
        "",
        "The intervals describe uncertainty across these sampled rooms for fixed models. They do not measure variation across training seeds or correct for repeated validation selection. Layouts can repeat within a split. Start rows are disjoint by construction: training uses 2/4/6, validation 1/3/5, test 7. This is a structured distribution shift, not an IID random split; a seed namespace alone would not have guaranteed different map contents.",
        "",
        "## Memory diagnostic",
        "",
        "A cue appears only in the first observation. All later observations, time, previous action, and inventory are identical across labels. The correct two-way choice depends on the cue. Each seed uses 2,000 synthetic examples, an 80/20 split, a one-layer transformer, and 10 supervised epochs.",
        "",
        "| Seed | History | Same model, history removed | Separately trained memoryless |",
        "| --- | --- | --- | --- |",
    ]
    for m in memories:
        lines.append(
            f"| {m['seed']} | {m[keys[0]]:.2%} | {m[keys[1]]:.2%} | {m[keys[2]]:.2%} |"
        )
    lines += [
        "",
        "![Memory diagnostic](../artifacts/charts/memory-probe.png)",
        "",
        "History removal changes the input distribution and leaves no label information; a constant prediction can be above or below 50% because the sampled labels are not exactly balanced. The independently trained memoryless baseline is the stronger control. These results establish cue access in this diagnostic, not memory dependence of the heist policy.",
        "",
        "## Failure modes and scope",
        "",
        "- The initial BC/PPO run regressed after warm-up and after its best checkpoint. Latest and champion are deliberately separate.",
        "- A short temporal window forgets observations older than 16 steps. The teacher can retain a map for the whole episode, which the student cannot exactly imitate in every state.",
        "- Demonstration distribution shift can trap the policy after an early wrong turn. PPO from scratch is sample-inefficient at this budget.",
        "- Camera maps add key/door dependencies and timing. Editor geometry checks cannot guarantee a safe timed route.",
        "- Showcase episodes are successful validation examples chosen for visualization; they are not a random sample or evidence of aggregate competence.",
        "- Only one main training seed per recipe was run. Stronger claims require repeated independent training seeds, larger validation suites, and a newly reserved test suite.",
        "- The automatic curriculum is implemented and tested for advancement and budget exhaustion. The reported camera transfer was launched explicitly, not presented as a completed autonomous mastery curriculum.",
        "",
        "## Reproduce the evidence",
        "",
        "Use the saved configs and commands in README. The camera command adds `--warm-start runs/banana-refined-seed11/champion.pt`. Run `python scripts/publish_study.py` after all recipes are frozen, then `python scripts/build_report.py`. Training is stochastic; exact values may differ across PyTorch versions, devices, and scheduling. Raw logs, configs, hashes, model weights, and the observation version travel with the repository.",
        "",
    ]
    (ROOT / "docs" / "EXPERIMENTS.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
