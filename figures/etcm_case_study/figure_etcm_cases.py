#!/usr/bin/env python3
"""Build the frozen ETCM2.0 representative-case manuscript figure."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


METHODS = ["Strict-HDCTI", "Hctx-P", "Hctx-P+CHCR"]
METHOD_LABELS = ["Strict", "Hctx-P", "Hctx-P\n+ CHCR"]
CASE_STYLES = {
    ("DEXPROPRANOLOL", "SIGMAR1"): {
        "color": "#2F6FB0",
        "marker": "o",
        "label": "DEXPROPRANOLOL–SIGMAR1",
    },
    ("Quercetin", "PLAU"): {
        "color": "#168C78",
        "marker": "s",
        "label": "Quercetin–PLAU",
    },
    ("Quercetin", "OPRD1"): {
        "color": "#C44E52",
        "marker": "X",
        "label": "Quercetin–OPRD1",
    },
}


def configure_matplotlib():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#222222",
        "xtick.color": "#444444",
        "ytick.color": "#444444",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "legend.frameon": False,
    })


def load_cases(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload["representative_cases"]
    if len(cases) != 3:
        raise ValueError("Expected exactly three frozen representative cases.")
    for case in cases:
        missing = [method for method in METHODS if method not in case["ranks"]]
        if missing:
            raise ValueError("Missing ranks for %s: %s" % (
                case["gene_symbol"], ", ".join(missing)
            ))
    return payload, cases


def write_source_data(path, cases):
    fields = [
        "validation_order",
        "compound_name",
        "gene_symbol",
        "uniprot_accession",
        "evidence_grade",
        "strict_rank",
        "hctx_p_rank",
        "hctx_p_chcr_rank",
        "best_rank_gain_vs_strict",
        "chdp_path_count",
        "external_evidence",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        for case in cases:
            writer.writerow({
                "validation_order": case["validation_order"],
                "compound_name": case["compound_name"],
                "gene_symbol": case["gene_symbol"],
                "uniprot_accession": case["uniprot_accession"],
                "evidence_grade": case["evidence_grade"],
                "strict_rank": case["ranks"]["Strict-HDCTI"],
                "hctx_p_rank": case["ranks"]["Hctx-P"],
                "hctx_p_chcr_rank": case["ranks"]["Hctx-P+CHCR"],
                "best_rank_gain_vs_strict": max(
                    case["rank_gain_vs_strict"].values()
                ),
                "chdp_path_count": case["chdp_path_count"],
                "external_evidence": case["external_evidence"],
            })


def panel_label(ax, label):
    ax.text(
        -0.10, 1.04, label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top",
        ha="left",
        color="#111111",
    )


def draw_rank_panel(ax, cases):
    x = list(range(len(METHODS)))
    for case in cases:
        key = (case["compound_name"], case["gene_symbol"])
        style = CASE_STYLES[key]
        ranks = [case["ranks"][method] for method in METHODS]
        linestyle = "--" if case["evidence_grade"] == "Conflict" else "-"
        ax.plot(
            x,
            ranks,
            color=style["color"],
            marker=style["marker"],
            markersize=5.2,
            markeredgewidth=0.8,
            linewidth=1.6,
            linestyle=linestyle,
            zorder=3,
        )
        for xi, rank in zip(x, ranks):
            ax.text(
                xi,
                rank - 0.55,
                str(rank),
                color=style["color"],
                fontsize=6.5,
                fontweight="bold",
                ha="center",
                va="bottom",
            )

    ax.set_xlim(-0.25, 2.45)
    ax.set_ylim(17.5, 0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(METHOD_LABELS)
    ax.set_yticks([1, 3, 5, 7, 10, 13, 16])
    ax.set_ylabel("Candidate rank (lower is better)")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.8)
    ax.tick_params(axis="x", length=0, pad=5)
    ax.tick_params(axis="y", length=3)
    ax.set_title(
        "Frozen candidate ranking",
        loc="left",
        fontsize=8,
        fontweight="bold",
        pad=8,
    )

    handles = []
    for case in cases:
        key = (case["compound_name"], case["gene_symbol"])
        style = CASE_STYLES[key]
        handles.append(Line2D(
            [0], [0],
            color=style["color"],
            marker=style["marker"],
            linestyle="--" if case["evidence_grade"] == "Conflict" else "-",
            linewidth=1.5,
            markersize=4.8,
            label=style["label"],
        ))
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.0, -0.10),
        fontsize=5.5,
        ncol=3,
        handlelength=1.8,
        borderaxespad=0,
        columnspacing=1.0,
    )
    panel_label(ax, "a")


def evidence_row(ax, y, color, compound, target, measurement, source, herb=None):
    ax.add_patch(Rectangle(
        (0.01, y - 0.17),
        0.98,
        0.29,
        transform=ax.transAxes,
        facecolor="#F7F9FB",
        edgecolor="#D8DEE6",
        linewidth=0.7,
        clip_on=False,
    ))
    ax.text(
        0.04, y + 0.055, compound,
        transform=ax.transAxes,
        fontsize=6.8,
        fontweight="bold",
        color="#222222",
        va="center",
    )
    ax.annotate(
        "",
        xy=(0.51, y + 0.055),
        xytext=(0.32, y + 0.055),
        xycoords=ax.transAxes,
        textcoords=ax.transAxes,
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4),
    )
    ax.text(
        0.53, y + 0.055, target,
        transform=ax.transAxes,
        fontsize=7.0,
        fontweight="bold",
        color=color,
        va="center",
    )
    ax.text(
        0.04, y - 0.035, measurement,
        transform=ax.transAxes,
        fontsize=6.2,
        color="#333333",
        va="center",
    )
    ax.text(
        0.04, y - 0.105, source,
        transform=ax.transAxes,
        fontsize=5.8,
        color="#666666",
        va="center",
    )
    if herb:
        ax.text(
            0.53, y - 0.035, herb,
            transform=ax.transAxes,
            fontsize=5.7,
            color="#6A5E40",
            va="center",
        )


def draw_evidence_panel(ax):
    ax.axis("off")
    ax.set_title(
        "Independent evidence for supported candidates",
        loc="left",
        fontsize=8,
        fontweight="bold",
        pad=8,
    )
    evidence_row(
        ax,
        0.73,
        "#2F6FB0",
        "DEXPROPRANOLOL",
        "SIGMAR1",
        "Ki 1,670 nM; IC50 3,974 nM",
        "Human radioligand-binding assay",
    )
    evidence_row(
        ax,
        0.34,
        "#168C78",
        "Quercetin",
        "PLAU",
        "IC50 12,100 nM + X-ray complex",
        "Human uPA assay; binding in the S1 pocket",
        herb="Herb context: JingDaJi",
    )
    ax.plot(
        [0.54, 0.93], [0.12, 0.12],
        transform=ax.transAxes,
        color="#8A7B55",
        linewidth=1.0,
        linestyle=(0, (3, 2)),
    )
    ax.text(
        0.54, 0.145, "8 post-hoc paths*",
        transform=ax.transAxes,
        fontsize=5.5,
        color="#6A5E40",
        va="bottom",
    )
    ax.text(
        0.04, 0.02,
        "*Database paths are hypotheses, not independent C–P validation.",
        transform=ax.transAxes,
        fontsize=5.6,
        color="#666666",
        va="bottom",
    )
    panel_label(ax, "b")


def draw_conflict_panel(ax):
    ax.axis("off")
    ax.set_title(
        "High rank does not guarantee activity",
        loc="left",
        fontsize=8,
        fontweight="bold",
        pad=8,
    )
    ax.add_patch(Rectangle(
        (0.01, 0.20),
        0.98,
        0.64,
        transform=ax.transAxes,
        facecolor="#FCF6F6",
        edgecolor="#E4C5C7",
        linewidth=0.8,
    ))
    ax.text(
        0.05, 0.70, "Quercetin",
        transform=ax.transAxes,
        fontsize=7.5,
        fontweight="bold",
        color="#222222",
        va="center",
    )
    ax.annotate(
        "",
        xy=(0.57, 0.70),
        xytext=(0.30, 0.70),
        xycoords=ax.transAxes,
        textcoords=ax.transAxes,
        arrowprops=dict(arrowstyle="-|>", color="#C44E52", lw=1.5),
    )
    ax.text(
        0.60, 0.70, "OPRD1",
        transform=ax.transAxes,
        fontsize=7.8,
        fontweight="bold",
        color="#C44E52",
        va="center",
    )
    ax.text(
        0.05, 0.51,
        "Model ranks: 3  /  4  /  3",
        transform=ax.transAxes,
        fontsize=6.6,
        color="#333333",
    )
    ax.text(
        0.05, 0.38,
        "Human binding assay: AC50 > 30,000 nM",
        transform=ax.transAxes,
        fontsize=6.6,
        fontweight="bold",
        color="#A43E43",
    )
    ax.text(
        0.05, 0.26,
        "Direct inactivity/very weak evidence (Conflict)",
        transform=ax.transAxes,
        fontsize=5.9,
        color="#666666",
    )
    ax.text(
        0.05, 0.05,
        "Solid arrows: direct experiment   ·   Dashed line: post-hoc path",
        transform=ax.transAxes,
        fontsize=5.5,
        color="#666666",
        va="bottom",
    )
    panel_label(ax, "c")


def build_figure(cases):
    width_in = 183 / 25.4
    height_in = 105 / 25.4
    fig = plt.figure(figsize=(width_in, height_in), facecolor="white")
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.05, 1.25],
        height_ratios=[1.15, 0.85],
        left=0.08,
        right=0.985,
        bottom=0.18,
        top=0.94,
        wspace=0.30,
        hspace=0.45,
    )
    ax_rank = fig.add_subplot(grid[:, 0])
    ax_evidence = fig.add_subplot(grid[0, 1])
    ax_conflict = fig.add_subplot(grid[1, 1])

    draw_rank_panel(ax_rank, cases)
    draw_evidence_panel(ax_evidence)
    draw_conflict_panel(ax_conflict)
    return fig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/etcm_topk_representative_cases.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures/etcm_case_study"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    configure_matplotlib()
    _, cases = load_cases(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    write_source_data(args.output_dir / "source_data.tsv", cases)
    fig = build_figure(cases)
    stem = args.output_dir / "etcm_representative_cases"
    fig.savefig(
        str(stem) + ".svg",
        format="svg",
        facecolor="white",
    )
    fig.savefig(
        str(stem) + ".pdf",
        format="pdf",
        facecolor="white",
        metadata={
            "Title": "ETCM2.0 representative compound-target cases",
            "Subject": "Frozen Top-K ranks and independent evidence",
        },
    )
    fig.savefig(
        str(stem) + ".png",
        format="png",
        dpi=400,
        facecolor="white",
    )
    plt.close(fig)
    print("Wrote %s.[svg|pdf|png]" % stem)
    print("Wrote %s" % (args.output_dir / "source_data.tsv"))


if __name__ == "__main__":
    main()
