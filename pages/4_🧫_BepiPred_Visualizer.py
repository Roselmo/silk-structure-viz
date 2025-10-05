# pages/4_🧫_BepiPred_Visualizer.py
from __future__ import annotations

from pathlib import Path
import io
import re
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt

# ---------------------------- Paths & constants ------------------------------
DATA_DIR   = Path("data")
TABLE_DIR  = DATA_DIR / "tables"
CSV_PATH   = TABLE_DIR / "epitopes_iedb.csv"

THRESHOLD_DEFAULT = 0.50  # BepiPred threshold for highlighting

# ---------------------------- Page header (EN) --------------------------------
st.title("BepiPred-2.0: Results — Sequential B-Cell Epitope Predict")

st.markdown(
    """
This page displays the **per-residue output** from BepiPred-2.0 for silk proteins.  
You can **select a protein**, view its **residue-level table**, and explore several
**interactive charts**. Points with **EpitopeProbability > 0.5** are highlighted and
can be downloaded as **TIFF (300 dpi)** for publication-quality figures.

*Reference:* Jespersen, M. C., Peters, B., Nielsen, M., & Marcatili, P. (2017). BepiPred-2.0: improving sequence-based B-cell epitope prediction using conformational epitopes. Nucleic acids research, 45(W1), W24-W29.
"""
)

# ---------------------------- Column normalization ---------------------------
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())

def _normalize_headers(cols) -> Dict[str, str]:
    """
    Map a variety of column names from BepiPred outputs to a stable schema:
      Protein, pos, aa, exposure, rsa, helix, sheet, coil, epiprob
    """
    mapping: Dict[str, str] = {}
    for c in cols:
        k = _norm(str(c))
        if k in {"protein","entry","name","antigen"}:
            mapping[c] = "Protein"
        elif k in {"pos","position","resindex","residue","resno"}:
            mapping[c] = "pos"
        elif k in {"aminoacid","aa","residueletter"}:
            mapping[c] = "aa"
        elif k in {"exposedburied","exposed_buried","exposure"}:
            mapping[c] = "exposure"
        elif k in {"relativesurfaceaccessilibity","relativesurfaceaccessibility","rsa"}:
            mapping[c] = "rsa"
        elif k in {"helixprobability","hprob","helix"}:
            mapping[c] = "helix"
        elif k in {"sheetprobability","eprob","sheet"}:
            mapping[c] = "sheet"
        elif k in {"coilprobability","cprob","coil"}:
            mapping[c] = "coil"
        elif k in {"epitopeprobability","bepipredscore","score","prob"}:
            mapping[c] = "epiprob"
        else:
            mapping[c] = c
    return mapping

def load_epitopes_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        st.error(f"File not found: `{path}`")
        st.stop()
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.read_csv(path, sep=None, engine="python")

    df = df.rename(columns=_normalize_headers(df.columns))

    # Basic coercions
    for num_col in ["pos", "rsa", "helix", "sheet", "coil", "epiprob"]:
        if num_col in df.columns:
            df[num_col] = pd.to_numeric(df[num_col], errors="coerce")

    # String cleanups
    if "exposure" in df.columns:
        df["exposure"] = df["exposure"].astype(str).str.upper().str.strip()
        # Normalizar a E/B/UNKNOWN
        df["exposure"] = df["exposure"].replace(
            {"EXPOSED":"E", "BURIED":"B", "U":"UNKNOWN", "NAN":"UNKNOWN"}
        )
        df.loc[~df["exposure"].isin(["E","B","UNKNOWN"]), "exposure"] = "UNKNOWN"

    # Ensure expected columns exist
    needed = ["Protein","pos","aa","exposure","rsa","helix","sheet","coil","epiprob"]
    for c in needed:
        if c not in df.columns:
            df[c] = np.nan

    # Drop completely empty rows
    df = df.dropna(how="all")
    return df

df_all = load_epitopes_csv(CSV_PATH)

# ---------------------------- UI controls ------------------------------------
proteins = sorted([p for p in df_all["Protein"].dropna().astype(str).unique()])
if not proteins:
    st.error("No proteins found in the CSV (column `Protein`).")
    st.stop()

col_sel, col_thr = st.columns([2,1])
with col_sel:
    protein = st.selectbox("Protein", proteins, index=0)
with col_thr:
    epi_thr = st.slider("Epitope threshold", 0.0, 1.0, THRESHOLD_DEFAULT, 0.01)

sub = df_all[df_all["Protein"].astype(str) == protein].copy()
sub = sub.sort_values("pos")

st.subheader("Residue table")
st.caption("Residue-level results for the selected antigen.")
st.dataframe(
    sub[["Protein","pos","aa","exposure","rsa","helix","sheet","coil","epiprob"]],
    use_container_width=True,
    hide_index=True,
)
# Table download
st.download_button(
    "Download table (CSV)",
    data=sub.to_csv(index=False).encode("utf-8"),
    file_name=f"{protein}_bepipred_residue.csv",
    mime="text/csv",
    use_container_width=True
)

# ---------------------------- Altair chart helpers ---------------------------
def alt_theme():
    alt.themes.enable("opaque")  # light background
alt_theme()

def alt_epiprob_line(df: pd.DataFrame, thr: float):
    base = alt.Chart(df).encode(x=alt.X("pos:Q", title="Position"))
    line = base.mark_line(color="#2F4B7C", opacity=0.9).encode(
        y=alt.Y("epiprob:Q", title="EpitopeProbability")
    )
    points = base.mark_point(filled=True, size=30).encode(
        y="epiprob:Q",
        color=alt.condition(
            alt.datum.epiprob > thr,
            alt.value("#6D2E8C"),  # purple
            alt.value("#2F4B7C")
        )
    )
    rule = alt.Chart(pd.DataFrame({"y":[thr]})).mark_rule(strokeDash=[4,4], color="#888").encode(y="y:Q")
    return (line + points + rule).properties(height=280)

def alt_scatter_prob(df: pd.DataFrame, ycol: str, ytitle: str, thr: float):
    base = alt.Chart(df).encode(x=alt.X("pos:Q", title="Position"))
    pts = base.mark_point(filled=True, size=28).encode(
        y=alt.Y(f"{ycol}:Q", title=ytitle),
        color=alt.condition(
            alt.datum.epiprob > thr,
            alt.value("#6D2E8C"),
            alt.value("#2F4B7C")
        )
    )
    return pts.properties(height=240)

def alt_bar_exposure(df: pd.DataFrame):
    # Count only E and B
    tmp = df[df["exposure"].isin(["E","B"])].copy()
    if tmp.empty:
        tmp = pd.DataFrame({"exposure":["E","B"], "count":[0,0]})
    else:
        tmp = tmp.groupby("exposure", as_index=False).size().rename(columns={"size":"count"})
    cat = alt.Chart(tmp).mark_bar().encode(
        x=alt.X("exposure:N", title="Exposed/Buried", sort=["E","B"]),
        y=alt.Y("count:Q", title="Count"),
        color=alt.value("#4C6EF5")
    )
    return cat.properties(height=200)

# ---------------------------- Matplotlib replicas (TIFF) ---------------------
def mpl_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.labelsize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
    })

def tiff_bytes_from_fig(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="tiff", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()

def mpl_epiprob_line(df: pd.DataFrame, thr: float) -> bytes:
    mpl_style()
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(df["pos"], df["epiprob"], color="#2F4B7C", linewidth=1.2)
    # Points colored by threshold
    hi = df["epiprob"] > thr
    ax.scatter(df.loc[~hi, "pos"], df.loc[~hi, "epiprob"], s=8, color="#2F4B7C", alpha=0.8)
    ax.scatter(df.loc[hi, "pos"], df.loc[hi, "epiprob"], s=10, color="#6D2E8C", alpha=0.85)
    ax.axhline(thr, color="#888", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Position")
    ax.set_ylabel("EpitopeProbability")
    ax.grid(True, axis="y", linestyle="--", alpha=0.25)
    for sp in ["top","right"]:
        ax.spines[sp].set_visible(False)
    return tiff_bytes_from_fig(fig)

def mpl_scatter_prob(df: pd.DataFrame, ycol: str, ytitle: str, thr: float) -> bytes:
    mpl_style()
    fig, ax = plt.subplots(figsize=(8, 3.0))
    hi = df["epiprob"] > thr
    ax.scatter(df.loc[~hi, "pos"], df.loc[~hi, ycol], s=8, color="#2F4B7C", alpha=0.8)
    ax.scatter(df.loc[hi, "pos"], df.loc[hi, ycol], s=10, color="#6D2E8C", alpha=0.9)
    ax.set_xlabel("Position")
    ax.set_ylabel(ytitle)
    ax.grid(True, axis="y", linestyle="--", alpha=0.25)
    for sp in ["top","right"]:
        ax.spines[sp].set_visible(False)
    return tiff_bytes_from_fig(fig)

def mpl_bar_exposure(df: pd.DataFrame) -> bytes:
    mpl_style()
    tmp = df[df["exposure"].isin(["E","B"])].copy()
    counts = tmp["exposure"].value_counts().reindex(["E","B"]).fillna(0).astype(int)
    fig, ax = plt.subplots(figsize=(5.0, 2.6))
    ax.bar(["E","B"], counts.values, color="#4C6EF5", width=0.6)
    ax.set_xlabel("Exposed/Buried")
    ax.set_ylabel("Count")
    for sp in ["top","right"]:
        ax.spines[sp].set_visible(False)
    ax.grid(True, axis="y", linestyle="--", alpha=0.25)
    return tiff_bytes_from_fig(fig)

# ---------------------------- SECTION: EpitopeProbability vs Position --------
st.subheader(f"{protein} — BepiPred scores")
chart_epi = alt_epiprob_line(sub, epi_thr)
st.altair_chart(chart_epi, use_container_width=True)

# Download (TIFF) for this chart
st.download_button(
    "Download TIFF (EpitopeProbability vs Position, 300 dpi)",
    data=mpl_epiprob_line(sub, epi_thr),
    file_name=f"{protein}_EpitopeProbability_vs_Position.tiff",
    mime="image/tiff",
    use_container_width=True
)

st.divider()

# ---------------------------- SECTION: Sequence tracks -----------------------
st.subheader("Sequence tracks")

# Exposure distribution (bar E/B)
exp_chart = alt_bar_exposure(sub)
st.altair_chart(exp_chart, use_container_width=True)
st.download_button(
    "Download TIFF (Exposure distribution, 300 dpi)",
    data=mpl_bar_exposure(sub),
    file_name=f"{protein}_Exposure_bar.tiff",
    mime="image/tiff",
    use_container_width=True
)

# RSA (scatter)
rsa_chart = alt_scatter_prob(sub, "rsa", "RSA", epi_thr)
st.altair_chart(rsa_chart, use_container_width=True)
st.download_button(
    "Download TIFF (RSA vs Position, 300 dpi)",
    data=mpl_scatter_prob(sub, "rsa", "RSA", epi_thr),
    file_name=f"{protein}_RSA_vs_Position.tiff",
    mime="image/tiff",
    use_container_width=True
)

# Helix prob (scatter)
helix_chart = alt_scatter_prob(sub, "helix", "HelixProbability", epi_thr)
st.altair_chart(helix_chart, use_container_width=True)
st.download_button(
    "Download TIFF (HelixProbability vs Position, 300 dpi)",
    data=mpl_scatter_prob(sub, "helix", "HelixProbability", epi_thr),
    file_name=f"{protein}_HelixProb_vs_Position.tiff",
    mime="image/tiff",
    use_container_width=True
)

# Sheet prob (scatter)
sheet_chart = alt_scatter_prob(sub, "sheet", "SheetProbability", epi_thr)
st.altair_chart(sheet_chart, use_container_width=True)
st.download_button(
    "Download TIFF (SheetProbability vs Position, 300 dpi)",
    data=mpl_scatter_prob(sub, "sheet", "SheetProbability", epi_thr),
    file_name=f"{protein}_SheetProb_vs_Position.tiff",
    mime="image/tiff",
    use_container_width=True
)

# Coil prob (scatter)
coil_chart = alt_scatter_prob(sub, "coil", "CoilProbability", epi_thr)
st.altair_chart(coil_chart, use_container_width=True)
st.download_button(
    "Download TIFF (CoilProbability vs Position, 300 dpi)",
    data=mpl_scatter_prob(sub, "coil", "CoilProbability", epi_thr),
    file_name=f"{protein}_CoilProb_vs_Position.tiff",
    mime="image/tiff",
    use_container_width=True
)