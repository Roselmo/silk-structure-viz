# pages/6_🧩_TCell_ClassI_Visualizer.py
from __future__ import annotations

from pathlib import Path
import io
import re
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ----------------------------- Paths & constants -----------------------------
DATA_DIR   = Path("data")
TABLE_DIR  = DATA_DIR / "tables"     # expects *_classI.tsv
CLASSI_SUFFIX = "_classI.tsv"

# ----------------------------- Intro (EN) ------------------------------------
st.title("NG-IEDB — MHC Class I: results & visualization")

st.markdown(
    """
**Next-generation (NG) IEDB tools** is a modern, fully redesigned edition of the IEDB Analysis Resource.
It offers an integrated environment to predict and analyze T- and B-cell epitopes with **fewer clicks,
cohesive pipelines** (e.g., predict → cluster → human similarity), and **interactive, shareable outputs**.
The **TC1 suite** consolidates MHC class I binding/elution, immunogenicity and antigen-processing predictors,
and scales to real-world datasets.

*Reference:* Yan, Z., Kim, K., Kim, H., Ha, B., Gambiez, A., Bennett, J., ... & Greenbaum, J. A. (2024). Next-generation IEDB tools: a platform for epitope prediction and analysis. Nucleic acids research, 52(W1), W526-W532.
"""
)

st.subheader("Table description")
st.markdown(
    """
This table contains **one row per peptide–allele prediction** produced by the TC1 suite.

- **seq** — identifier of the source protein sequence.  
- **peptide** — amino-acid sequence of the evaluated peptide.  
- **start / end** — zero-based positions of the peptide within the source sequence.  
- **peptide length** — peptide size (aa).  
- **allele** — HLA class I allele used for the prediction (e.g., *HLA-A*02:01*).  
- **peptide index** — row identifier supplied by the tool.  
- **median binding percentile** — summary percentile across methods (**lower implies stronger binders**).  
- **netmhcpan_el score / percentile** — elution-based presentation predictor (percentile is relative to random; **lower is better**).  
- **netmhcpan_ba IC50 / percentile** — binding-affinity predictor (smaller **IC50** / percentile indicates stronger binders).  
- **netmhcpan_el core / icore; netmhcpan_ba core / icore** — predicted 9-mer binding cores (not plotted below).  
- **immunogenicity score** — likelihood that a presented peptide elicits a T-cell response.  
- **proteasome / TAP / MHC / processing / processing total scores** — components and aggregate of antigen processing and presentation.

Use the selector below to choose a silk protein, inspect the rows, and export the table.
Then explore **per-allele distributions** with publication-ready boxplots.
"""
)

# ----------------------------- Column normalization --------------------------
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())

def _normalize_headers(cols) -> Dict[str, str]:
    mapping = {}
    for c in cols:
        key = _norm(str(c))
        if key in {"seq","sequence","protein","source"}:
            mapping[c] = "seq"
        elif key in {"peptide"}:
            mapping[c] = "peptide"
        elif key in {"start","begin"}:
            mapping[c] = "start"
        elif key in {"end","stop"}:
            mapping[c] = "end"
        elif key in {"peptidelength","length","len"}:
            mapping[c] = "peptide length"
        elif key in {"allele","hla","mhc"}:
            mapping[c] = "allele"
        elif key in {"peptideindex","idx","index"}:
            mapping[c] = "peptide index"
        elif key in {"medianbindingpercentile","medianpercentile","medianbindpct"}:
            mapping[c] = "median binding percentile"

        # NetMHCpan EL (class I)
        elif key in {"netmhcpanelscore","netmhcpanelutionscore","elutionscore"}:
            mapping[c] = "netmhcpan_el score"
        elif key in {"netmhcpanelpercentile","netmhcpanelutionpercentile","elutionpercentile"}:
            mapping[c] = "netmhcpan_el percentile"
        elif key in {"netmhcpanelcore"}:
            mapping[c] = "netmhcpan_el core"
        elif key in {"netmhcpanelicore"}:
            mapping[c] = "netmhcpan_el icore"

        # NetMHCpan BA (class I)
        elif key in {"netmhcpanbacore"}:
            mapping[c] = "netmhcpan_ba core"
        elif key in {"netmhcpanbaicore"}:
            mapping[c] = "netmhcpan_ba icore"
        elif key in {"netmhcpanbaic50","netmhcpanic50","ic50","ic50nm"}:
            mapping[c] = "netmhcpan_ba IC50"
        elif key in {"netmhcpanbapercentile"}:
            mapping[c] = "netmhcpan_ba percentile"

        # pathway & immunogenicity
        elif key in {"immunogenicityscore","immunoscore"}:
            mapping[c] = "immunogenicity score"
        elif key in {"proteasomescore","proteasome"}:
            mapping[c] = "proteasome score"
        elif key in {"tapscore","tap"}:
            mapping[c] = "tap score"
        elif key in {"mhcscore","mhc"}:
            mapping[c] = "mhc score"
        elif key in {"processingscore"}:
            mapping[c] = "processing score"
        elif key in {"processingtotalscore","totalprocessingscore","totalscore"}:
            mapping[c] = "processing total score"
        else:
            mapping[c] = c
    return mapping

# ----------------------------- Data loading -----------------------------------
def list_classI_files() -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    if not TABLE_DIR.exists():
        return out
    for p in TABLE_DIR.glob(f"*{CLASSI_SUFFIX}"):
        name = p.name[:-len(CLASSI_SUFFIX)]
        out[name] = p
    return dict(sorted(out.items(), key=lambda kv: kv[0].lower()))

def load_classI_table(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, sep="\t")
    except Exception:
        df = pd.read_csv(path, sep=None, engine="python")
    df = df.rename(columns=_normalize_headers(df.columns))

    # Ensure expected columns exist
    for must in ["seq","peptide","start","end","peptide length","allele"]:
        if must not in df.columns:
            df[must] = np.nan

    # Convert numerics
    numeric_cols = [
        "median binding percentile",
        "netmhcpan_el score", "netmhcpan_el percentile",
        "netmhcpan_ba IC50", "netmhcpan_ba percentile",
        "immunogenicity score", "proteasome score", "tap score",
        "mhc score", "processing score", "processing total score",
        "start","end","peptide length",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# ----------------------------- Plot helpers (Matplotlib) ----------------------
def _apply_pub_style(ax: plt.Axes):
    ax.grid(True, axis="y", linestyle="--", alpha=0.25)
    ax.grid(False, axis="x")
    for spine in ["top","right"]:
        ax.spines[spine].set_visible(False)

def make_fig_boxplot_with_points(labels: List[str], groups: List[np.ndarray], pretty_ylabel: str) -> plt.Figure:
    """
    Publication style: outline-only boxplots + small jittered points; no title
    (the section header acts as the title). The figure shown is exactly what is downloaded.
    """
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.labelsize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 10,
        "figure.dpi": 300,
        "savefig.dpi": 300,
    })

    width = max(6, min(16, 0.30 * max(1, len(labels))))
    fig, ax = plt.subplots(figsize=(width, 4.2))

    # Boxplot (outline only; no fill)
    ax.boxplot(
        groups, patch_artist=True, labels=labels,
        boxprops=dict(facecolor="none", edgecolor="#33415C", linewidth=1.2),
        medianprops=dict(color="#33415C", linewidth=1.6),
        whiskerprops=dict(color="#33415C", linewidth=1.0),
        capprops=dict(color="#33415C", linewidth=1.0),
        flierprops=dict(markeredgecolor="#33415C", markersize=2, alpha=0.15),
    )

    # Small jittered points
    rng = np.random.default_rng(42)
    for i, vals in enumerate(groups, start=1):
        if vals.size == 0:
            continue
        x = i + rng.normal(0.0, 0.06, size=vals.size)
        ax.scatter(x, vals, s=4, color="#2F4B7C", alpha=0.45, linewidths=0)

    ax.set_xlabel("Allele")
    ax.set_ylabel(pretty_ylabel)
    _apply_pub_style(ax)
    plt.xticks(rotation=90, ha="center", va="top")
    plt.tight_layout()
    return fig

def prepare_groups_per_allele(df: pd.DataFrame, col: str) -> Tuple[List[str], List[np.ndarray]]:
    if "allele" not in df.columns or col not in df.columns:
        return [], []
    d = pd.DataFrame({"allele": df["allele"], "_y": pd.to_numeric(df[col], errors="coerce")}).dropna()
    if d.empty:
        return [], []
    labels, groups = [], []
    for a, sub in d.groupby("allele", sort=True):
        vals = sub["_y"].values.astype(float)
        if vals.size:
            labels.append(a)
            groups.append(vals)
    return labels, groups

def render_and_download(df: pd.DataFrame, col: str, pretty_y_label: str, protein: str):
    labels, groups = prepare_groups_per_allele(df, col)
    if not labels:
        st.info(f"Column **{col}** not found or not plottable.")
        return
    fig = make_fig_boxplot_with_points(labels, groups, pretty_y_label)
    st.pyplot(fig, use_container_width=True)
    buf = io.BytesIO()
    fig.savefig(buf, format="tiff", dpi=300)
    st.download_button(
        label="Download TIFF (300 dpi)",
        data=buf.getvalue(),
        file_name=f"{protein}_{re.sub('[^A-Za-z0-9]+','_',pretty_y_label)}.tiff",
        mime="image/tiff",
        use_container_width=True
    )
    plt.close(fig)

# ----------------------------- UI: file pick, table, charts -------------------
files = list_classI_files()
if not files:
    st.warning(f"No Class I TSV files were found in `{TABLE_DIR}` (expected pattern `*_classI.tsv`).")
    st.stop()

st.subheader("Choose protein")
protein = st.selectbox("Protein", list(files.keys()), index=0)
df_all = load_classI_table(files[protein])

st.subheader("Filtered table")
st.caption("One row per peptide–allele prediction from NG-IEDB TC1.")
show_cols = [c for c in [
    "seq","peptide","start","end","peptide length","allele","peptide index",
    "median binding percentile",
    "netmhcpan_el score","netmhcpan_el percentile",
    "netmhcpan_ba IC50","netmhcpan_ba percentile",
    "immunogenicity score","proteasome score","tap score","mhc score",
    "processing score","processing total score"
] if c in df_all.columns]
st.dataframe(df_all[show_cols], use_container_width=True, hide_index=True)

csv_bytes = df_all[show_cols].to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", csv_bytes, file_name=f"{protein}_classI_results.csv",
                   mime="text/csv", use_container_width=True)

# ----------------------------- Charts panel -----------------------------------
st.subheader("Per-allele distributions")

# NOTE: removed BA core/icore plots; removed “(lower is better)” from axis labels.
plots = [
    ("netmhcpan_el score",      "NetMHCpan-EL score"),
    ("netmhcpan_el percentile", "NetMHCpan-EL percentile"),
    ("netmhcpan_ba IC50",       "NetMHCpan-BA IC50 (nM)"),
    ("netmhcpan_ba percentile", "NetMHCpan-BA percentile"),
    ("immunogenicity score",    "Immunogenicity score"),
    ("proteasome score",        "Proteasome score"),
    ("tap score",               "TAP score"),
    ("mhc score",               "MHC score"),
    ("processing score",        "Processing score"),
    ("processing total score",  "Processing total score"),
]

for col, pretty_label in plots:
    st.markdown(f"**{pretty_label}**")
    render_and_download(df_all, col, pretty_label, protein)
    st.divider()