# pages/7_🔬_TCell_ClassII_Visualizer.py
from __future__ import annotations

from pathlib import Path
import io
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ----------------------------- Paths & constants -----------------------------
DATA_DIR   = Path("data")
TABLE_DIR  = DATA_DIR / "tables"     # expects *_classII.tsv
CLASSII_SUFFIX = "_classII.tsv"

# ----------------------------- Intro (EN) ------------------------------------
st.title("NG-IEDB — MHC Class II: Results and Visualization")

st.markdown(
    """
The **Next-generation (NG) IEDB tools** provide a unified, modern interface for **T- and B-cell epitope prediction**.  
For **MHC Class II (TC2)**, the platform integrates **NetMHCIIpan** presentation/binding predictions with **MHCII-NP**
cleavage and motif models, enabling full end-to-end workflows with large-scale capability and interactive visualization.  

*Reference:*Yan, Z., Kim, K., Kim, H., Ha, B., Gambiez, A., Bennett, J., ... & Greenbaum, J. A. (2024). Next-generation IEDB tools: a platform for epitope prediction and analysis. Nucleic acids research, 52(W1), W526-W532.
"""
)

st.subheader("Table description")
st.markdown(
    """
This dataset corresponds to **predictions from the TC2 suite** of NG-IEDB tools:

- **seq** — identifier of the source protein sequence.  
- **peptide** — amino acid sequence of the peptide evaluated.  
- **start / end** — peptide start and end positions within the parent protein.  
- **peptide length** — number of amino acids in the peptide.  
- **allele** — MHC class II allele (e.g., *HLA-DRB1*01:01*).  
- **peptide index** — numeric identifier of the prediction row.  
- **median binding percentile** — summary percentile across predictors (lower = stronger binding).  
- **NetMHCIIpan-EL score / percentile** — likelihood of presentation (lower percentile = higher confidence).  
- **MHCII-NP cleavage score / percentile** — predicted cleavage probability from the MHCII-NP model.  
"""
)

# ----------------------------- Header normalization --------------------------
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())

def _normalize_headers(cols) -> Dict[str, str]:
    m = {}
    for c in cols:
        k = _norm(str(c))
        if k in {"seq","sequence","protein","source"}:
            m[c] = "seq"
        elif k == "peptide":
            m[c] = "peptide"
        elif k in {"start","begin"}:
            m[c] = "start"
        elif k in {"end","stop"}:
            m[c] = "end"
        elif k in {"peptidelength","length","len"}:
            m[c] = "peptide length"
        elif k in {"allele","hla","mhc"}:
            m[c] = "allele"
        elif k in {"peptideindex","idx","index"}:
            m[c] = "peptide index"
        elif k in {"medianbindingpercentile","medianpercentile"}:
            m[c] = "median binding percentile"
        elif k in {"netmhciipanelcore","netmhciipan_core"}:
            m[c] = "netmhciipan_el core"
        elif k in {"netmhciipanelscore","netmhciipan_elscore"}:
            m[c] = "netmhciipan_el score"
        elif k in {"netmhciipanpercentile","netmhciipan_elpercentile"}:
            m[c] = "netmhciipan_el percentile"
        elif k in {"mhchiinpcleavageprobabilityscore"}:
            m[c] = "MHCII-NP Cleavage probability score"
        elif k in {"mhchiinpcleavageprobabilitypercentilerank"}:
            m[c] = "MHCII-NP Cleavage probability percentile rank"
        else:
            m[c] = c
    return m

# ----------------------------- Data loading -----------------------------------
def list_classII_files() -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    if not TABLE_DIR.exists():
        return out
    for p in TABLE_DIR.glob(f"*{CLASSII_SUFFIX}"):
        name = p.name[:-len(CLASSII_SUFFIX)]
        out[name] = p
    return dict(sorted(out.items(), key=lambda kv: kv[0].lower()))

def load_classII_table(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, sep="\t")
    except Exception:
        df = pd.read_csv(path, sep=None, engine="python")
    df = df.rename(columns=_normalize_headers(df.columns))

    must_have = [
        "seq","peptide","start","end","peptide length","allele","peptide index",
        "median binding percentile","netmhciipan_el score","netmhciipan_el percentile",
        "MHCII-NP Cleavage probability score","MHCII-NP Cleavage probability percentile rank",
    ]
    for col in must_have:
        if col not in df.columns:
            df[col] = np.nan

    numeric_cols = [
        "start","end","peptide length",
        "median binding percentile",
        "netmhciipan_el score","netmhciipan_el percentile",
        "MHCII-NP Cleavage probability score","MHCII-NP Cleavage probability percentile rank",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# ----------------------------- Plot helpers -----------------------------------
def _apply_pub_style(ax: plt.Axes):
    ax.grid(True, axis="y", linestyle="--", alpha=0.25)
    ax.grid(False, axis="x")
    for s in ["top","right"]:
        ax.spines[s].set_visible(False)

def make_fig_boxplot_with_points(labels: List[str], groups: List[np.ndarray], pretty_ylabel: str) -> plt.Figure:
    """Elegant scientific-style plot: outline-only boxes + small jittered points."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
    })
    width = max(6, min(16, 0.3 * max(1, len(labels))))
    fig, ax = plt.subplots(figsize=(width, 4.2))

    ax.boxplot(
        groups, patch_artist=True, labels=labels,
        boxprops=dict(facecolor="none", edgecolor="#33415C", linewidth=1.1),
        medianprops=dict(color="#33415C", linewidth=1.3),
        whiskerprops=dict(color="#33415C", linewidth=1.0),
        capprops=dict(color="#33415C", linewidth=1.0),
        flierprops=dict(markeredgecolor="#33415C", markersize=1.5, alpha=0.15),
    )

    rng = np.random.default_rng(42)
    for i, vals in enumerate(groups, start=1):
        if vals.size == 0:
            continue
        x = i + rng.normal(0.0, 0.06, size=vals.size)
        ax.scatter(x, vals, s=3, color="#2F4B7C", alpha=0.45, linewidths=0)

    ax.set_xlabel("Allele", labelpad=6)
    # Break long y-axis names automatically
    wrapped = "\n".join(pretty_ylabel.split())
    ax.set_ylabel(wrapped, labelpad=8)
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
files = list_classII_files()
if not files:
    st.warning(f"No Class II TSV files were found in `{TABLE_DIR}` (expected pattern `*_classII.tsv`).")
    st.stop()

st.subheader("Choose protein")
protein = st.selectbox("Protein", list(files.keys()), index=0)
df_all = load_classII_table(files[protein])

st.subheader("Filtered table")
show_cols = [c for c in [
    "seq","peptide","start","end","peptide length","allele","peptide index",
    "median binding percentile","netmhciipan_el score","netmhciipan_el percentile",
    "MHCII-NP Cleavage probability score","MHCII-NP Cleavage probability percentile rank",
] if c in df_all.columns]
st.dataframe(df_all[show_cols], use_container_width=True, hide_index=True)

csv_bytes = df_all[show_cols].to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", csv_bytes, file_name=f"{protein}_classII_results.csv",
                   mime="text/csv", use_container_width=True)

# ----------------------------- Charts panel -----------------------------------
st.subheader("Per-allele distributions")

plots = [
    ("netmhciipan_el score", "NetMHCIIpan-EL score"),
    ("netmhciipan_el percentile", "NetMHCIIpan-EL percentile"),
    ("MHCII-NP Cleavage probability score", "MHCII-NP cleavage probability score"),
    ("MHCII-NP Cleavage probability percentile rank", "MHCII-NP cleavage percentile rank"),
    ("median binding percentile", "Median binding percentile"),
]

for col, pretty_label in plots:
    if col in df_all.columns:
        st.markdown(f"**{pretty_label}**")
        render_and_download(df_all, col, pretty_label, protein)
        st.divider()