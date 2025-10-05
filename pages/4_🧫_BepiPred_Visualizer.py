# pages/4_🧫_BepiPred_Visualizer.py
from __future__ import annotations

from pathlib import Path
import re
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt

# ---------- Config ----------
alt.data_transformers.disable_max_rows()

DATA_DIR   = Path("data")
TABLE_DIR  = DATA_DIR / "tables"
CSV_PATH   = TABLE_DIR / "epitopes_iedb.csv"

THRESHOLD = 0.5  # BepiPred default cutoff

# ---------- Small helpers ----------
def _normalize_headers(cols: list[str]) -> list[str]:
    """lowercase + drop spaces/accents/dashes/underscores for forgiving column matching."""
    tr = str.maketrans("óáéíúñ", "oaeiun")
    out = []
    for c in cols:
        s = str(c).translate(tr).lower()
        s = re.sub(r"[ \-_/]", "", s)
        out.append(s)
    return out

def _pick(norm_cols: list[str], candidates: set[str]) -> int | None:
    """pick first exact (or prefix/suffix) match index for any candidate token"""
    for i, nc in enumerate(norm_cols):
        if nc in candidates:
            return i
    for i, nc in enumerate(norm_cols):
        if any(nc.startswith(c) or nc.endswith(c) for c in candidates):
            return i
    return None

def load_bepipred_csv(csv_path: Path) -> pd.DataFrame:
    """Load a per-residue BepiPred table and standardize column names.

    Expected columns (with robust synonyms accepted):
      Protein:   Entry / Protein / Name / Antigen / SequenceName
      Position:  Position / Pos / Residue / Resid
      AminoAcid: AminoAcid / AA
      Exposed/Buried: Exposed/Buried / Exposure / State
      RelativeSurfaceAccessilibity: RSA / RelativeSurfaceAccessilibity / RelativeSurfaceAccessibility
      HelixProbability: HelixProbability / Helix
      SheetProbability: SheetProbability / Sheet / Beta
      CoilProbability:  CoilProbability / Coil
      EpitopeProbability: EpitopeProbability / BepiPred score / Score / Probability
    """
    if not csv_path.exists():
        st.error(f"CSV not found: `{csv_path}`")
        st.stop()

    try:
        df = pd.read_csv(csv_path, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(csv_path)

    orig = list(df.columns.astype(str))
    norm = _normalize_headers(orig)

    # define synonym sets
    prot_syn = {"entry","protein","name","antigen","sequencename","seqname"}
    pos_syn  = {"position","pos","residue","resid","resindex"}
    aa_syn   = {"aminoacid","aa","resname"}
    exp_syn  = {"exposedburied","exposed","buried","exposure","state"}
    rsa_syn  = {"relativesurfaceaccessilibity","relativesurfaceaccessibility","rsa"}
    hel_syn  = {"helixprobability","helix","alphaprobability"}
    sht_syn  = {"sheetprobability","sheet","betaprobability"}
    coil_syn = {"coilprobability","coil"}
    prb_syn  = {"epitopeprobability","bepipredscore","score","probability","prob","bepipred2score"}

    i_prot = _pick(norm, prot_syn)
    i_pos  = _pick(norm, pos_syn)
    i_aa   = _pick(norm, aa_syn)
    i_exp  = _pick(norm, exp_syn)
    i_rsa  = _pick(norm, rsa_syn)
    i_hel  = _pick(norm, hel_syn)
    i_sht  = _pick(norm, sht_syn)
    i_coil = _pick(norm, coil_syn)
    i_prb  = _pick(norm, prb_syn)

    required = {"Protein": i_prot, "Position": i_pos, "AminoAcid": i_aa, "EpitopeProbability": i_prb}
    if any(v is None for v in required.values()):
        st.error(
            "The CSV is missing required columns. "
            "Minimum needed: Entry/Protein, Position, AminoAcid, EpitopeProbability."
        )
        st.stop()

    # build standardized dataframe
    colmap = {
        orig[i_prot]: "Protein",
        orig[i_pos]:  "Position",
        orig[i_aa]:   "AminoAcid",
        orig[i_prb]:  "EpitopeProbability",
    }
    if i_exp is not None:  colmap[orig[i_exp]] = "Exposed/Buried"
    if i_rsa is not None:  colmap[orig[i_rsa]] = "RelativeSurfaceAccessilibity"
    if i_hel is not None:  colmap[orig[i_hel]] = "HelixProbability"
    if i_sht is not None:  colmap[orig[i_sht]] = "SheetProbability"
    if i_coil is not None: colmap[orig[i_coil]] = "CoilProbability"

    out = df.rename(columns=colmap)

    # type cleanup
    out["Protein"] = out["Protein"].astype(str)
    out["Position"] = pd.to_numeric(out["Position"], errors="coerce").astype("Int64")
    out["EpitopeProbability"] = pd.to_numeric(out["EpitopeProbability"], errors="coerce")

    for opt in ["RelativeSurfaceAccessilibity","HelixProbability","SheetProbability","CoilProbability"]:
        if opt in out.columns:
            out[opt] = pd.to_numeric(out[opt], errors="coerce")

    # Basic ordering
    out = out.sort_values(["Protein","Position"], kind="stable").reset_index(drop=True)
    return out

def protein_selector(df: pd.DataFrame) -> str:
    prots = sorted(df["Protein"].dropna().astype(str).unique(), key=str.lower)
    return st.selectbox("Protein", prots, index=0)

def download_filtered_table(df: pd.DataFrame, filename: str):
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download CSV",
        data=csv_bytes,
        file_name=filename,
        mime="text/csv",
        use_container_width=True
    )

# ---------- Charts ----------
def chart_prob_vs_pos(df: pd.DataFrame):
    d = df[["Position","EpitopeProbability"]].dropna()
    if d.empty:
        st.info("No EpitopeProbability values to plot for this protein.")
        return
    d["Above"] = d["EpitopeProbability"] > THRESHOLD

    line = alt.Chart(d).mark_line().encode(
        x=alt.X("Position:Q", title="Position"),
        y=alt.Y("EpitopeProbability:Q", title="EpitopeProbability", scale=alt.Scale(domain=[0,1])),
        tooltip=[alt.Tooltip("Position:Q"), alt.Tooltip("EpitopeProbability:Q", format=".3f")]
    ).properties(height=300)

    pts = alt.Chart(d).mark_circle(size=36).encode(
        x="Position:Q",
        y="EpitopeProbability:Q",
        color=alt.condition("datum.Above", alt.value("purple"), alt.value("#3465a4")),
        tooltip=[alt.Tooltip("Position:Q"), alt.Tooltip("EpitopeProbability:Q", format=".3f")]
    )

    rule = alt.Chart(pd.DataFrame({"y":[THRESHOLD]})).mark_rule(color="#999", strokeDash=[6,3]).encode(y="y:Q")
    st.altair_chart((line + pts + rule).interactive(), use_container_width=True)

def chart_exposed_bar(df: pd.DataFrame):
    """Bar chart counting Exposed (E) and Buried (B) residues."""
    if "Exposed/Buried" not in df.columns:
        st.info("Exposed/Buried column is not available in this CSV.")
        return

    d = df.copy()

    # Normalize to 'Exposed' or 'Buried' from a variety of inputs: 'E', 'B', 'exposed', 'buried', etc.
    def _norm_exposure(val: object) -> str:
        s = str(val).strip().lower()
        if not s:
            return "Unknown"
        # look at first letter or full word
        if s[0] == "e" or s in {"exposed","expose","exterior","surface","expuesto"}:
            return "Exposed"
        if s[0] == "b" or s in {"buried","interior","core","enterrado"}:
            return "Buried"
        return "Unknown"

    d["Exposure"] = d["Exposed/Buried"].apply(_norm_exposure)

    # Keep only Exposed/Buried categories for the bar chart; drop Unknowns to avoid a single large bar
    counts = (
        d[d["Exposure"].isin(["Exposed","Buried"])]
        .value_counts("Exposure")
        .reset_index(name="Count")
    )

    # Ensure both bars appear even if one is zero
    for cat in ["Exposed","Buried"]:
        if cat not in counts["Exposure"].values:
            counts = pd.concat([counts, pd.DataFrame({"Exposure":[cat], "Count":[0]})], ignore_index=True)

    counts = counts.sort_values("Exposure")  # alphabetical Exposed, Buried

    chart = alt.Chart(counts).mark_bar().encode(
        x=alt.X("Exposure:N", title="Exposed/Buried", sort=["Exposed","Buried"]),
        y=alt.Y("Count:Q", title="Count"),
        color=alt.Color("Exposure:N", scale=alt.Scale(range=["#6a9ae2", "#8892b0"])),
        tooltip=["Exposure","Count"]
    ).properties(height=220)
    st.altair_chart(chart, use_container_width=True)

def _chart_value_vs_pos(df: pd.DataFrame, value_col: str, title_y: str):
    d = df[["Position", value_col, "EpitopeProbability"]].dropna()
    if d.empty:
        st.info(f"{value_col} is not available to plot for this protein.")
        return
    d["Above"] = d["EpitopeProbability"] > THRESHOLD

    line = alt.Chart(d).mark_line().encode(
        x=alt.X("Position:Q", title="Position"),
        y=alt.Y(f"{value_col}:Q", title=title_y),
        tooltip=[alt.Tooltip("Position:Q"), alt.Tooltip(f"{value_col}:Q", format=".3f")]
    ).properties(height=260)

    pts = alt.Chart(d).mark_circle(size=36).encode(
        x="Position:Q",
        y=f"{value_col}:Q",
        color=alt.condition("datum.Above", alt.value("purple"), alt.value("#3465a4")),
        tooltip=[alt.Tooltip("Position:Q"), alt.Tooltip(f"{value_col}:Q", format=".3f"),
                 alt.Tooltip("EpitopeProbability:Q", format=".3f")]
    )
    st.altair_chart((line + pts).interactive(), use_container_width=True)

# ---------- UI ----------
st.title("BepiPred-2.0: Results Sequential B-Cell Epitope Predict")

st.markdown(
    """
**Overview.** We analyzed silk proteins with **BepiPred-2.0**, a sequence-based predictor of **linear B-cell epitopes**.
In short, the tool takes a protein sequence and estimates, for each residue, the probability of belonging to a B-cell epitope.
This helps flag surface-accessible segments that are more likely to be recognized by antibodies—useful for vaccine design, antibody engineering, and biomaterials assessment.

**What BepiPred-2.0 does (in brief):**
- Uses a **random-forest** model trained on high-quality antibody–antigen 3D complexes.
- Returns **per-residue scores** (EpitopeProbability). By convention, residues with **score > 0.5** are considered predicted epitope positions.
- Includes auxiliary features from structure-informed predictors (e.g., **exposure** and **secondary-structure propensities**) that support interpretation.

Below you can select a silk protein, inspect the table, download it, and explore interactive charts summarizing
**EpitopeProbability**, **exposure**, **relative surface accessibility**, and **secondary-structure propensities**.
"""
)

# Load data
df_all = load_bepipred_csv(CSV_PATH)

# Protein selection
st.subheader("Select protein and view table")
protein = protein_selector(df_all)
df = df_all[df_all["Protein"].astype(str).str.fullmatch(protein, case=False, na=False)].copy()
if df.empty:
    # fallback: contains
    token = re.sub(r"[^a-z0-9]", "", protein.lower())
    df = df_all[df_all["Protein"].str.lower().str.replace(r"[^a-z0-9]","", regex=True).str.contains(token)]

# Show table + download
st.dataframe(
    df[
        [c for c in ["Protein","Position","AminoAcid","Exposed/Buried",
                     "RelativeSurfaceAccessilibity","HelixProbability",
                     "SheetProbability","CoilProbability","EpitopeProbability"]
         if c in df.columns]
    ],
    use_container_width=True,
    hide_index=True
)
download_filtered_table(df, f"{protein}_bepipred.csv")

# Charts
st.subheader("EpitopeProbability vs Position")
chart_prob_vs_pos(df)

st.subheader("Exposure distribution")
chart_exposed_bar(df)

st.subheader("Relative Surface Accessibility (RSA) vs Position")
if "RelativeSurfaceAccessilibity" in df.columns:
    _chart_value_vs_pos(df, "RelativeSurfaceAccessilibity", "Relative Surface Accessibility")
else:
    st.info("RelativeSurfaceAccessilibity column is not available in this CSV.")

st.subheader("HelixProbability vs Position")
if "HelixProbability" in df.columns:
    _chart_value_vs_pos(df, "HelixProbability", "Helix Probability")
else:
    st.info("HelixProbability column is not available in this CSV.")

st.subheader("SheetProbability vs Position")
if "SheetProbability" in df.columns:
    _chart_value_vs_pos(df, "SheetProbability", "Sheet (β) Probability")
else:
    st.info("SheetProbability column is not available in this CSV.")

st.subheader("CoilProbability vs Position")
if "CoilProbability" in df.columns:
    _chart_value_vs_pos(df, "CoilProbability", "Coil Probability")
else:
    st.info("CoilProbability column is not available in this CSV.")