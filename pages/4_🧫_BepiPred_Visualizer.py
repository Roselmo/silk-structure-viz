# pages/4_🧫_BepiPred_Visualizer.py
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# ---------------- Altair setup ----------------
alt.data_transformers.disable_max_rows()
alt.themes.enable("opaque")

# ---------------- Helpers ----------------
def first_key(d: dict, candidates: list[str]) -> str | None:
    """Return the first key in candidates that exists in dict d (case-insensitive)."""
    kl = {k.lower(): k for k in d.keys()}
    for c in candidates:
        if c.lower() in kl:
            return kl[c.lower()]
    return None

def _to_list_like(x, N: int | None = None, fill=""):
    """
    Normalize a field to a list:
      - list -> return as-is
      - str  -> list of non-space characters
      - else -> list of length N filled with `fill` (if N provided)
    """
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        s = "".join(ch for ch in x if not ch.isspace())
        return list(s)
    if N is not None:
        return [fill] * N
    return []

def load_bepipred_json(path: str | Path) -> dict:
    with open(path, "r") as fh:
        return json.load(fh)

def antigen_records(payload: dict) -> dict:
    """Support {'antigens': {...}} or a flat dict of antigens."""
    if isinstance(payload, dict) and isinstance(payload.get("antigens"), dict):
        return payload["antigens"]
    return payload

def to_dataframe(name: str, rec: dict) -> pd.DataFrame:
    """
    Build tidy DF robust to key names & string-encoded tracks.
    Keys candidates:
      AA:   AA/aa/sequence/Sequence
      PRED: PRED/pred/score/scores/bepipred/BepiPred
      EXP:  EXPOSED/Exposure/exposed/Surface/surface
      RSA:  RSA/rsa/ASA/asa
    """
    k_aa   = first_key(rec, ["AA", "aa", "sequence", "Sequence"])
    k_pred = first_key(rec, ["PRED", "pred", "score", "scores", "bepipred", "BepiPred"])
    if not (k_aa and k_pred):
        raise ValueError(f"Antigen '{name}' missing AA or PRED arrays.")

    aa   = _to_list_like(rec[k_aa])
    pred = _to_list_like(rec[k_pred])
    N = len(aa)
    if len(pred) != N:
        raise ValueError(f"'{name}': AA ({N}) and PRED ({len(pred)}) length mismatch.")

    k_exp = first_key(rec, ["EXPOSED", "Exposure", "exposed", "Surface", "surface"])
    k_rsa = first_key(rec, ["RSA", "rsa", "ASA", "asa"])

    exp = _to_list_like(rec.get(k_exp, []), N=N, fill="")
    rsa = rec.get(k_rsa, [np.nan]*N)
    rsa = _to_list_like(rsa, N=N, fill=np.nan)

    df = pd.DataFrame({
        "pos": np.arange(1, N+1, dtype=int),
        "aa": aa,
        "pred": pd.to_numeric(pd.Series(pred), errors="coerce"),
        "exp": exp,
        "rsa": pd.to_numeric(pd.Series(rsa), errors="coerce")
    })
    # pos2 for per-residue bars (x2 must NOT carry a type in Altair v5)
    df["pos2"] = df["pos"] + 1

    # Normalize RSA to 0..1 if it looks like 0..100
    if df["rsa"].notna().any():
        mx = df["rsa"].max(skipna=True)
        if pd.notna(mx) and mx > 1.5:
            df["rsa"] = df["rsa"] / 100.0

    # Exposure labels
    exp_map = {"E": "Exposed", "B": "Buried"}
    df["exp_label"] = df["exp"].map(exp_map).fillna("Unknown")

    return df

def contiguous_segments(mask: pd.Series) -> pd.DataFrame:
    """Return 1-based start/end for contiguous True runs."""
    arr = mask.values.astype(int)
    edges = np.diff(np.r_[0, arr, 0])
    starts = np.where(edges == 1)[0] + 1
    ends   = np.where(edges == -1)[0]
    return pd.DataFrame({"start": starts, "end": ends})

# ---------------- UI ----------------
st.title("BepiPred-2.0: Sequential B-Cell Epitope Predictor.")

# Intro (solo la primera línea que querías conservar)
st.markdown(
    """
**Prediction of potential linear B-cell epitopes.**
    """
)

json_path = Path("data/tables/results.json")
if not json_path.exists():
    st.error(f"JSON file not found at `{json_path}`.")
    st.stop()

payload = load_bepipred_json(json_path)
antigens = antigen_records(payload)
names = list(antigens.keys())
if not names:
    st.error("No antigens found in the JSON.")
    st.stop()

left, right = st.columns([1, 2])
with left:
    name = st.selectbox("Antigen", names, index=0)
with right:
    thr = st.slider("Epitope threshold (BepiPred score)", 0.0, 1.0, 0.5, 0.01)

rec = antigens[name]
try:
    df = to_dataframe(name, rec)
except Exception as e:
    st.exception(e)
    st.stop()

df["above"] = (df["pred"] >= thr)
segments = contiguous_segments(df["above"])

# ---------------- Main chart ----------------
base = alt.Chart(df).properties(width=1050, height=260)

area = base.mark_area(opacity=0.18, color="#4C78A8").encode(
    x="pos:Q",
    y=alt.Y("pred:Q", scale=alt.Scale(domain=[0,1]), title="BepiPred score"),
)

line = base.mark_line(strokeWidth=2, color="#305E96").encode(
    x=alt.X("pos:Q", title="Position"),
    y="pred:Q",
    tooltip=[
        alt.Tooltip("pos:Q", title="Position"),
        alt.Tooltip("aa:N", title="AA"),
        alt.Tooltip("pred:Q", title="Score", format=".3f"),
        alt.Tooltip("exp_label:N", title="Exposure"),
        alt.Tooltip("rsa:Q", title="RSA", format=".2f"),
    ],
)

thr_rule = alt.Chart(pd.DataFrame({"thr":[thr]})).mark_rule(
    strokeDash=[6,4], color="#333"
).encode(y="thr:Q")

if len(segments):
    seg_chart = alt.Chart(segments).mark_rect(
        opacity=0.25, color="#FFB000"
    ).encode(
        x=alt.X("start", title=None),
        x2="end",
        y=alt.value(0), y2=alt.value(1)
    ).properties(height=260)
    top = seg_chart + area + line + thr_rule
else:
    top = area + line + thr_rule

st.subheader(f"{name} — BepiPred scores")
st.altair_chart(top, use_container_width=True)

# ---------------- Sequence tracks (Exposure + RSA) ----------------
st.subheader("Sequence tracks")

# Exposure track
exp_colors = alt.Scale(
    domain=["Exposed", "Buried", "Unknown"],
    range=["#6DBD45", "#9E77B0", "#E0E0E0"]
)
exp_track = alt.Chart(df).mark_rect(height=14).encode(
    x=alt.X("pos:Q", title=None),
    x2="pos2",  # Altair v5: x2 must be plain field
    color=alt.Color("exp_label:N", scale=exp_colors, legend=alt.Legend(title="Exposure")),
    tooltip=[
        alt.Tooltip("pos:Q", title="Position"),
        alt.Tooltip("aa:N", title="AA"),
        alt.Tooltip("exp_label:N", title="Exposure")
    ]
).properties(height=34, width=1050)

# RSA (line + area)
rsa_base = alt.Chart(df).properties(width=1050, height=140)
rsa_area = rsa_base.mark_area(opacity=0.2, color="#4C78A8").encode(
    x=alt.X("pos:Q", title="pos"),
    y=alt.Y("rsa:Q", title="RSA", scale=alt.Scale(domain=[0,1]))
)
rsa_line = rsa_base.mark_line(strokeWidth=2, color="#305E96").encode(
    x="pos:Q", y="rsa:Q",
    tooltip=[alt.Tooltip("pos:Q"), alt.Tooltip("aa:N"), alt.Tooltip("rsa:Q", format=".2f")]
)

tracks = alt.vconcat(exp_track, (rsa_area + rsa_line)).resolve_scale(color="independent")
st.altair_chart(tracks, use_container_width=True)

# ---------------- Data table with dynamic filters ----------------
st.subheader("Data table")

with st.expander("Filters", expanded=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        pos_min, pos_max = int(df["pos"].min()), int(df["pos"].max())
        pos_range = st.slider("Position range", pos_min, pos_max, (pos_min, pos_max))
        only_above = st.checkbox("Only residues above threshold", value=False)
    with c2:
        score_min, score_max = float(df["pred"].min()), float(df["pred"].max())
        score_range = st.slider("Score range", 0.0, 1.0, (max(0.0, score_min), min(1.0, score_max)), 0.01)
        exp_opts = st.multiselect("Exposure", options=sorted(df["exp_label"].unique()),
                                  default=list(sorted(df["exp_label"].unique())))
    with c3:
        rsa_present = df["rsa"].notna().any()
        if rsa_present:
            rsa_min = float(np.nanmin(df["rsa"]))
            rsa_max = float(np.nanmax(df["rsa"]))
            rsa_range = st.slider("RSA range", 0.0, 1.0, (max(0.0, rsa_min), min(1.0, rsa_max)), 0.01)
        else:
            rsa_range = None
        motif = st.text_input("AA motif (regex)", value="")

# Apply filters
mask = (
    (df["pos"] >= pos_range[0]) &
    (df["pos"] <= pos_range[1]) &
    (df["pred"] >= score_range[0]) &
    (df["pred"] <= score_range[1]) &
    (df["exp_label"].isin(exp_opts))
)
if only_above:
    mask &= df["above"]

if rsa_range is not None:
    mask &= df["rsa"].between(rsa_range[0], rsa_range[1]) | df["rsa"].isna()

if motif.strip():
    try:
        mask &= df["aa"].str.contains(motif, regex=True, na=False)
    except Exception:
        st.warning("Invalid regex in 'AA motif'. Showing unfiltered AA.")
        # keep mask as is

df_filt = df.loc[mask].copy()

# Show filtered table (non-editable, sortable)
st.dataframe(df_filt, use_container_width=True, hide_index=True)

# Download filtered CSV
csv = df_filt.to_csv(index=False).encode("utf-8")
st.download_button(
    f"Download {name} filtered table (CSV)",
    csv,
    file_name=f"{name}_bepipred_table_filtered.csv",
    mime="text/csv"
)

# ---------------- KPIs ----------------
cols = st.columns(3)
with cols[0]:
    st.metric("Sequence length", f"{len(df)} aa")
with cols[1]:
    st.metric("Segments ≥ threshold", f"{len(segments)}")
with cols[2]:
    total_len = int((segments["end"] - segments["start"] + 1).sum()) if len(segments) else 0
    st.metric("Total epitope length", f"{total_len} aa")

# ---------------- Guide ----------------
st.markdown(
    """
**What the charts show**

- **BepiPred scores**: per-residue scores (0–1). The dashed line is the threshold; orange bands mark contiguous segments above it (putative **linear B-cell epitopes**).
- **Exposure track**: residues predicted as **Exposed** vs **Buried** (if present). Exposed segments are more likely to be antibody-accessible.
- **RSA**: **Relative Solvent Accessibility** (0..1)—higher means more surface exposure.

Tip: adjust the **threshold** to widen/narrow predicted epitopes and use the **Filters** above the table to subset residues of interest. Then export via **Download**.
    """
)