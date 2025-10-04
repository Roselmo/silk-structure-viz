# pages/6_🧩_TCell_ClassI_Visualizer.py
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

alt.data_transformers.disable_max_rows()
alt.themes.enable("opaque")

# ---------- helpers ----------
def first_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    m = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in m:
            return m[c.lower()]
    return None

def to_numeric(df: pd.DataFrame, cols: list[str]):
    for c in cols:
        if c and c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def log10_safe(x: pd.Series) -> pd.Series:
    # avoid log of 0 or negative
    return np.log10(np.maximum(x.astype(float), 1e-9))

def safe_range(label, vmin, vmax, default=None, step=1.0, key=None):
    if vmin is None or vmax is None or not np.isfinite(vmin) or not np.isfinite(vmax):
        st.caption(f"{label}: no numeric data available.")
        return None
    if vmin < vmax:
        if default is None:
            default = (vmin, vmax)
        return st.slider(label, vmin, vmax, default, step=step, key=key)
    else:
        st.caption(f"{label}: all values are {vmin}.")
        return (vmin, vmax)

# ---------- UI ----------
st.title("T-cell Epitope Analytics — MHC Class I")

st.markdown("""
This page summarizes **MHC Class I** binding predictions at scientific depth.  
We visualize:
- **IC50 (nM)**: predicted affinity (lower is stronger). We also plot **log₁₀(IC50)** for better distribution shapes.
- **Percentile rank**: rank-normalized binding score (lower is stronger).
- **Peptide length** (typically 8–11 aa for Class I).
- **Binder classes** (Strong/Moderate/Weak) using common IC50 cutoffs.
If start–end positions are present, we also show **positional coverage** across the protein.
""")

path = Path("data/tables/t_cellPrediction_ClassI.tsv")
if not path.exists():
    st.error("File not found: data/tables/t_cellPrediction_ClassI.tsv")
    st.stop()

df = pd.read_csv(path, sep="\t")

# Column detection (robust to IEDB variants)
col_pep   = first_col(df, ["peptide","Peptide","sequence","Sequence"])
col_len   = first_col(df, ["peptide length","Length"])
col_alle  = first_col(df, ["allele","Allele"])
col_ic50  = first_col(df, ["IC50","ic50","netmhcpan_ba IC50","ic50(nM)"])
col_pct   = first_col(df, ["percentile","median binding percentile","netmhcpan_ba percentile","netmhcpan_el percentile"])
col_start = first_col(df, ["start","Start"])
col_end   = first_col(df, ["end","End"])

# numeric coercion
to_numeric(df, [col_len, col_ic50, col_pct, col_start, col_end])

# Derived fields
if col_ic50:
    df["log10_IC50"] = log10_safe(df[col_ic50])
if col_len is None and col_pep:
    df["peptide_length"] = df[col_pep].astype(str).str.len()
    col_len = "peptide_length"

# Binder class (typical thresholds)
# Strong: < 50 nM; Moderate: 50–500 nM; Weak: > 500 nM
if col_ic50:
    def binder_class(x):
        if pd.isna(x): return "Unknown"
        if x < 50: return "Strong (<50 nM)"
        if x <= 500: return "Moderate (50–500 nM)"
        return "Weak (>500 nM)"
    df["binder_class"] = df[col_ic50].apply(binder_class)

# -------- filters --------
with st.expander("Filters", expanded=True):
    c1, c2 = st.columns([1,1])
    with c1:
        alleles = sorted(df[col_alle].astype(str).unique()) if col_alle else []
        sel_alleles = st.multiselect("Alleles", alleles, default=alleles or None, key="cls1_alleles")
    with c2:
        if col_ic50:
            vmin, vmax = df[col_ic50].min(skipna=True), df[col_ic50].max(skipna=True)
            _ = safe_range("IC50 (nM)", float(vmin) if pd.notna(vmin) else None,
                           float(vmax) if pd.notna(vmax) else None,
                           step=1.0, key="cls1_ic50")
        # (we still let all values through; charts will show full range—use table filters if needed)

mask = pd.Series(True, index=df.index)
if col_alle and sel_alleles:
    mask &= df[col_alle].astype(str).isin(sel_alleles)
df_f = df.loc[mask].copy()

# -------- charts --------
st.subheader("Affinity distributions")

charts = []

if col_ic50:
    # Histogram of log10(IC50)
    hist = alt.Chart(df_f).mark_bar().encode(
        x=alt.X("log10_IC50:Q", title="log₁₀(IC50 nM)"),
        y=alt.Y("count()", title="Count"),
        tooltip=[alt.Tooltip("count()", title="Count")]
    ).properties(width=520, height=260)
    charts.append(hist)

if col_pct:
    # Histogram of percentile
    hist_pct = alt.Chart(df_f).mark_bar().encode(
        x=alt.X(f"{col_pct}:Q", title="Percentile rank (lower = stronger)"),
        y=alt.Y("count()", title="Count"),
        tooltip=[alt.Tooltip("count()", title="Count")]
    ).properties(width=520, height=260)
    charts.append(hist_pct)

if charts:
    st.altair_chart(alt.hconcat(*charts), use_container_width=True)

# Per-allele spread (box-like via rule+points)
if col_alle and col_ic50:
    st.subheader("Per-allele affinity (log₁₀(IC50))")
    base = alt.Chart(df_f).encode(
        x=alt.X(f"{col_alle}:N", title="Allele", sort="-y")
    ).properties(height=320)

    points = base.mark_circle(size=35, opacity=0.35).encode(
        y=alt.Y("log10_IC50:Q", title="log₁₀(IC50 nM)"),
        tooltip=[col_alle, col_pep if col_pep else alt.value(""), alt.Tooltip("log10_IC50:Q", format=".2f")]
    )
    # Median line per allele
    med = base.mark_rule(size=3).encode(
        y=alt.Y("median(log10_IC50):Q", title=""),
        tooltip=[alt.Tooltip("median(log10_IC50):Q", title="Median log₁₀(IC50)", format=".2f")]
    )
    st.altair_chart(points + med, use_container_width=True)

# Binder class counts per allele
if col_alle and col_ic50:
    st.subheader("Binder classes by allele")
    cls_order = ["Strong (<50 nM)", "Moderate (50–500 nM)", "Weak (>500 nM)", "Unknown"]
    bar = alt.Chart(df_f).mark_bar().encode(
        x=alt.X(f"{col_alle}:N", title="Allele", sort="-y"),
        y=alt.Y("count()", title="Count"),
        color=alt.Color("binder_class:N", scale=alt.Scale(scheme="tableau10"), sort=cls_order, title="Class"),
        tooltip=[col_alle, alt.Tooltip("count()", title="Count")]
    ).properties(height=320)
    st.altair_chart(bar, use_container_width=True)

# Positional coverage (if available)
if col_start and col_end:
    st.subheader("Positional coverage (if start/end provided)")
    cov = alt.Chart(df_f).mark_rect(opacity=0.5).encode(
        x=alt.X(f"{col_start}:Q", title="Start"),
        x2=alt.X2(f"{col_end}:Q"),
        y=alt.Y(f"{col_alle}:N", title="Allele"),
        tooltip=[col_alle, col_pep if col_pep else alt.value(""), col_start, col_end]
    ).properties(height=320)
    st.altair_chart(cov, use_container_width=True)

# -------- table & download --------
st.subheader("Filtered data")
st.dataframe(df_f, use_container_width=True, hide_index=True)

csv = df_f.to_csv(index=False).encode("utf-8")
st.download_button("Download filtered Class I table (CSV)", csv, file_name="classI_filtered.csv", mime="text/csv")

# -------- column explanations --------
st.markdown("""
### Column meanings (Class I)
- **Peptide**: amino-acid subsequence evaluated for binding.
- **Allele**: specific MHC Class I molecule (e.g., HLA-A\*02:01).
- **IC50 (nM)**: predicted binding affinity; **lower** values indicate **stronger** binding.
- **log₁₀(IC50)**: log-transformed affinity to stabilize wide ranges and reveal structure in the distribution.
- **Percentile rank**: normalized binding score; **lower** percentiles indicate **stronger** predicted binders across an allele-specific background model.
- **Peptide length**: number of residues in the peptide (Class I is typically 8–11 aa).
- **Binder class**: categorical interpretation of IC50 (Strong <50 nM; Moderate 50–500 nM; Weak >500 nM).
- **Start / End**: peptide position in the source protein; used to display **positional coverage** when available.
""")