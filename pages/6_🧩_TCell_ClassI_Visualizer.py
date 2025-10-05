# pages/6_🧩_TCell_ClassI_Visualizer.py
from pathlib import Path
import re
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

alt.data_transformers.disable_max_rows()
alt.themes.enable("opaque")

DATA_DIR = Path("data/tables")

# ----------------- helpers -----------------
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
    return np.log10(np.maximum(pd.to_numeric(x, errors="coerce").astype(float), 1e-9))

def list_class_files(kind: str) -> list[Path]:
    # kind: "classI" or "classII"
    return sorted(DATA_DIR.glob(f"*_{kind}.tsv"))

def load_selected_files(files: list[Path]) -> pd.DataFrame:
    frames = []
    for p in files:
        try:
            df = pd.read_csv(p, sep="\t")
            # protein name from filename prefix (e.g., FibH_classI.tsv -> FibH)
            prot = re.sub(r"_classI\.tsv$", "", p.name, flags=re.IGNORECASE)
            df["Protein"] = prot
            frames.append(df)
        except Exception as e:
            st.warning(f"Skipping {p.name}: {e}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

# ----------------- UI -----------------
st.title("T-cell Epitope Analytics — MHC Class I")

st.markdown("""
This page summarizes **MHC Class I** binding predictions across split tables by protein.  
Use the selectors below to pick the protein table(s) and alleles; the plots update interactively.

**Scientific meaning of plotted columns**
- **IC50 (nM)**: predicted binding affinity; **lower** values indicate **stronger** binding. We use **log₁₀(IC50)** to stabilize wide ranges.
- **Percentile rank**: rank-normalized binding score per allele; **lower** percentiles indicate **stronger** binders.
Results are visualized per **allele** using **boxplots** (with points) to highlight spread and outliers.
""")

# ---- choose protein tables ----
all_files = list_class_files("classI")
if not all_files:
    st.error("No Class I split tables found in `data/tables/*_classI.tsv`.")
    st.stop()

protein_options = ["All proteins"] + [re.sub(r"_classI\.tsv$", "", p.name, flags=re.IGNORECASE) for p in all_files]
sel = st.multiselect("Select protein tables (Class I)", protein_options, default=["All proteins"])

if "All proteins" in sel or not sel:
    files_to_load = all_files
else:
    names = set(sel)
    files_to_load = [p for p in all_files if re.sub(r"_classI\.tsv$", "", p.name, flags=re.IGNORECASE) in names]

df = load_selected_files(files_to_load)
if df.empty:
    st.warning("No data loaded for the selected proteins.")
    st.stop()

# ---- column detection ----
col_pep   = first_col(df, ["peptide","Peptide","sequence","Sequence"])
col_alle  = first_col(df, ["allele","Allele"])
col_ic50  = first_col(df, ["IC50","ic50","netmhcpan_ba IC50","ic50(nM)"])
col_pct   = first_col(df, ["percentile","median binding percentile","netmhcpan_ba percentile","netmhcpan_el percentile"])

to_numeric(df, [col_ic50, col_pct])

if col_ic50:
    df["log10_IC50"] = log10_safe(df[col_ic50])

# ---- filters ----
alleles = sorted(df[col_alle].astype(str).unique()) if col_alle else []
sel_alleles = st.multiselect("Alleles", alleles, default=alleles or None, key="clsI_alleles")

mask = pd.Series(True, index=df.index)
if col_alle and sel_alleles:
    mask &= df[col_alle].astype(str).isin(sel_alleles)
df_f = df.loc[mask].copy()

# ---- metric selector ----
metrics = []
if col_ic50: metrics.append("log10(IC50)")
if col_pct:  metrics.append("Percentile")
if not metrics:
    st.error("No IC50 or Percentile columns detected.")
    st.stop()

metric = st.radio("Select metric for boxplot", metrics, horizontal=True)

# ---- boxplot ----
if metric == "log10(IC50)":
    y_encoding = alt.Y("log10_IC50:Q", title="log₁₀(IC50 nM)")
else:
    y_encoding = alt.Y(f"{col_pct}:Q", title="Percentile (lower = stronger)")

bp = alt.Chart(df_f).transform_filter(
    alt.datum[ col_alle if col_alle else "allele" ] != None
).mark_boxplot(extent="min-max").encode(
    x=alt.X(f"{col_alle}:N", title="Allele", sort="-y"),
    y=y_encoding,
    color=alt.Color("Protein:N", title="Protein"),
    tooltip=[ "Protein", col_alle ] + ([col_pep] if col_pep else [])
).properties(height=380)

pts = alt.Chart(df_f).mark_circle(size=28, opacity=0.35).encode(
    x=alt.X(f"{col_alle}:N", title="Allele", sort="-y"),
    y=y_encoding,
    color=alt.Color("Protein:N", title="Protein"),
    tooltip=[ "Protein", col_alle ] + ([col_pep] if col_pep else [])
)

st.altair_chart(bp + pts, use_container_width=True)

# ---- table & download ----
st.subheader("Filtered data")
st.dataframe(df_f, use_container_width=True, hide_index=True)
st.download_button(
    "Download filtered Class I table (CSV)",
    df_f.to_csv(index=False).encode("utf-8"),
    file_name="classI_filtered.csv",
    mime="text/csv",
)