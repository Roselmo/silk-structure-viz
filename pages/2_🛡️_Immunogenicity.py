# pages/2_🛡️_Immunogenicity.py
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

# ------------------ helpers ------------------
SEQ_MAP = {
    "1": "FibH", "2": "FibL", "3": "P25",
    "4": "Sericina_1", "5": "Sericina_2",
    "6": "Sericina_3", "7": "Sericina_4",
}

def first_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in cols:
            return cols[c.lower()]
    return None

def coerce_num(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c and c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# ------------------ page ------------------
st.title("Epitope Prediction and Immunogenicity Analysis")

st.markdown("""
The predictions shown here were generated with the **IEDB Analysis Resource**, a toolbox that brings together state-of-the-art methods for epitope discovery and interpretation, and complements the curated Immune Epitope Database (IEDB).  
Within this suite you’ll find tools aimed at **B-cell epitopes** (regions likely to be antibody-accessible) and **T-cell epitopes** (driven by MHC binding and peptide processing).  
The platform is actively maintained — for example, the **v2.28 update (Apr 2024)** integrated **NetMHCIIpan 4.2 and 4.3** into class-II binding along with bug fixes (including improved Axel-F support), improving accuracy and consistency across workflows.
""")

# =================================================================
# B-cell Epitope Prediction
# =================================================================
st.header("B-cell Epitope Prediction")

st.markdown("""
We used **BepiPred-2.0: Sequential B-Cell Epitope Predictor** to identify **linear** antibody-accessible segments across the silk protein set — **FIHC, FILC, P25, and Sericin 1–4**.  
These predictions are informative for assessing **immunogenicity** and **biocompatibility** when designing silk-based biomaterials.
""")

bcell_path = Path("data/tables/epitopes_iedb.csv")
if not bcell_path.exists():
    st.warning("B-cell epitope table not found at `data/tables/epitopes_iedb.csv`.")
else:
    dfb = pd.read_csv(bcell_path)

    # Si el CSV trae 'Seq' 1..7 podemos mapear a nombre legible,
    # pero NO mostramos Protein/Length en la tabla final.
    seq_col = first_col(dfb, ["Seq", "seq", "Sequence", "sequence"])
    if seq_col and "Protein" not in dfb.columns:
        dfb["Protein"] = dfb[seq_col].astype(str).map(SEQ_MAP).fillna(dfb[seq_col].astype(str))

    # Ocultar columnas no solicitadas
    drop_cols = [c for c in dfb.columns if c.lower() in {"protein", "length", "len"}]
    dfb_show = dfb.drop(columns=drop_cols, errors="ignore")

    st.subheader("B-cell epitope table (BepiPred-2.0)")
    st.dataframe(dfb_show, use_container_width=True, hide_index=True)

    st.download_button(
        "Download B-cell epitopes (CSV)",
        dfb_show.to_csv(index=False).encode("utf-8"),
        file_name="bcell_epitopes.csv",
        mime="text/csv",
    )

# =================================================================
# T-cell Epitope Prediction and MHC Binding
# =================================================================
st.header("T-cell Epitope Prediction and MHC Binding")

st.markdown("""
To characterize potential **T-cell** responses, we relied on IEDB’s **MHC binding predictors**.  
These models estimate how tightly a peptide is expected to bind a specific MHC molecule (reported as **IC50** or **percentile**).  
Binding is a **required** step for T-cell recognition — not a guarantee on its own — but it sharply narrows plausible candidates.

- **Class I** looks at short peptides (≈8–11 aa) for presentation by MHC-I.  
- **Class II** targets longer peptides and uses consensus strategies (NN-align, SMM-align, combinatorial libraries; the IEDB **v2.28** backend integrates **NetMHCIIpan 4.2/4.3**) for robust predictions.
""")

tables_dir = Path("data/tables")
file_I  = tables_dir / "t_cellPrediction_ClassI.tsv"
file_II = tables_dir / "t_cellPrediction_ClassII.tsv"

# ---------- Class I ----------
st.subheader("MHC Class I binding predictions")
if not file_I.exists():
    st.warning("Class I table not found at `data/tables/t_cellPrediction_ClassI.tsv`.")
else:
    dfi = pd.read_csv(file_I, sep="\t")

    i_alle  = first_col(dfi, ["allele", "Allele"])
    # Filtro ÚNICO: alelos
    with st.expander("Filters (Class I)", expanded=True):
        alleles = sorted(dfi[i_alle].astype(str).unique()) if i_alle else []
        sel_alleles = st.multiselect("Alleles", alleles, default=alleles, key="alleles_I")

    mask_i = pd.Series(True, index=dfi.index)
    if i_alle and sel_alleles:
        mask_i &= dfi[i_alle].astype(str).isin(sel_alleles)

    dfi_f = dfi.loc[mask_i].copy()
    st.dataframe(dfi_f, use_container_width=True, hide_index=True)
    st.download_button(
        "Download filtered Class I table (CSV)",
        dfi_f.to_csv(index=False).encode("utf-8"),
        file_name="tcell_classI_filtered.csv",
        mime="text/csv",
    )

# ---------- Class II ----------
st.subheader("MHC Class II binding predictions")
if not file_II.exists():
    st.warning("Class II table not found at `data/tables/t_cellPrediction_ClassII.tsv`.")
else:
    dfii = pd.read_csv(file_II, sep="\t")

    ii_alle  = first_col(dfii, ["allele", "Allele"])
    # Filtro ÚNICO: alelos (key distinto para evitar IDs duplicados)
    with st.expander("Filters (Class II)", expanded=True):
        alleles2 = sorted(dfii[ii_alle].astype(str).unique()) if ii_alle else []
        sel_alleles2 = st.multiselect("Alleles", alleles2, default=alleles2, key="alleles_II")

    mask_ii = pd.Series(True, index=dfii.index)
    if ii_alle and sel_alleles2:
        mask_ii &= dfii[ii_alle].astype(str).isin(sel_alleles2)

    dfii_f = dfii.loc[mask_ii].copy()
    st.dataframe(dfii_f, use_container_width=True, hide_index=True)
    st.download_button(
        "Download filtered Class II table (CSV)",
        dfii_f.to_csv(index=False).encode("utf-8"),
        file_name="tcell_classII_filtered.csv",
        mime="text/csv",
    )