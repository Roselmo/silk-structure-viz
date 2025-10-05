# pages/4_🧫_BepiPred_Visualizer.py
from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd
import streamlit as st
import py3Dmol
import streamlit.components.v1 as components
import altair as alt
import shlex

# ---- Altair config -----------------------------------------------------------
alt.data_transformers.disable_max_rows()
alt.themes.enable("opaque")

# ---- Paths -------------------------------------------------------------------
DATA_DIR = Path("data")
TABLE_DIR = DATA_DIR / "tables"
STRUCT_DIR = DATA_DIR / "structures"
CSV_PATH = TABLE_DIR / "epitopes_iedb.csv"

# ---- Small string helpers ----------------------------------------------------
def _norm_token(s: str) -> str:
    """lowercase + only [a-z0-9] for matching names."""
    return re.sub(r"[^a-z0-9]", "", str(s).lower())

def _norm_series_tokens(s: pd.Series) -> pd.Series:
    """Vectorized normalization for Series."""
    return s.astype(str).str.lower().str.replace(r"[^a-z0-9]", "", regex=True)

def _name_aliases(protein: str) -> list[str]:
    """Flexible aliases (case-insensitive, remove underscores, Sericina→Sericin)."""
    p = str(protein).strip()
    aliases = {p, p.lower(), p.upper(), p.capitalize(), p.replace("_", "")}
    m = re.match(r"^sericina[_\s-]?(\d+)$", p, flags=re.IGNORECASE)
    if m:
        num = m.group(1)
        aliases.add(f"Sericin{num}")
        aliases.add(f"sericin{num}")
    if re.match(r"^sericina$", p, flags=re.IGNORECASE):
        aliases.add("Sericin")
        aliases.add("sericin")
    aliases |= {a.replace("_", "") for a in aliases}
    # preserve order
    return list(dict.fromkeys(aliases))

# ---- Find structure file -----------------------------------------------------
def find_structure_for(protein: str) -> Path | None:
    if not STRUCT_DIR.exists():
        return None
    all_files = (
        list(STRUCT_DIR.glob("*.cif")) + list(STRUCT_DIR.glob("*.mmcif")) + list(STRUCT_DIR.glob("*.pdb")) +
        list(STRUCT_DIR.glob("*/*.cif")) + list(STRUCT_DIR.glob("*/*.mmcif")) + list(STRUCT_DIR.glob("*/*.pdb"))
    )
    if not all_files:
        return None

    aliases = _name_aliases(protein)
    aliases_norm = {_norm_token(a) for a in aliases}

    exact, partial = [], []
    for p in all_files:
        stem_norm = _norm_token(p.stem)
        if stem_norm in aliases_norm:
            exact.append(p)
        elif any(stem_norm in _norm_token(a) or _norm_token(a) in stem_norm for a in aliases):
            partial.append(p)

    candidates = exact if exact else partial
    if not candidates:
        return None

    def _rank(q: Path) -> tuple[int, int]:
        sfx = q.suffix.lower()
        if sfx in (".cif", ".mmcif"): return (0, len(q.name))
        if sfx == ".pdb": return (1, len(q.name))
        return (2, len(q.name))

    return sorted(set(candidates), key=_rank)[0]

def _compact_ranges(indices: list[int]) -> str:
    """[1,2,3,7,10,11] -> '1-3,7,10-11' (py3Dmol 'resi' selector)."""
    if not indices:
        return ""
    xs = sorted(set(int(x) for x in indices))
    start = prev = xs[0]
    out = []
    for x in xs[1:]:
        if x == prev + 1:
            prev = x
        else:
            out.append((start, prev))
            start = prev = x
    out.append((start, prev))
    return ",".join([f"{a}-{b}" if a != b else f"{a}" for a, b in out])

# ---- mmCIF / PDB residue-map helpers (robust) --------------------------------
def _parse_cif_atom_site(text: str):
    """Extract (chain, label_seq_id[int], auth_seq_id[int]) from _atom_site loop. Tolerant to wrapped rows."""
    try:
        lines = text.splitlines()
        rows, cols = [], []
        in_loop = False
        for i, ln in enumerate(lines):
            s = ln.strip()
            if s.lower() == "loop_":
                in_loop = True
                cols = []
                continue
            if in_loop and s.startswith("_"):
                cols.append(s)
                continue
            if in_loop:
                # is this an atom_site loop?
                if not cols or not any(c.startswith("_atom_site.") for c in cols):
                    in_loop = False
                    continue
                try:
                    idx_label = [j for j, c in enumerate(cols) if c.endswith("label_seq_id")][0]
                    idx_auth  = [j for j, c in enumerate(cols) if c.endswith("auth_seq_id")][0]
                    idx_chain = [j for j, c in enumerate(cols) if c.endswith("auth_asym_id")][0]
                except IndexError:
                    in_loop = False
                    continue

                j = i
                while j < len(lines):
                    l = lines[j].strip()
                    if not l or l.startswith("_") or l.lower() == "loop_" or l.lower().startswith("data_"):
                        in_loop = False
                        break
                    toks = shlex.split(l)
                    k = 1
                    while len(toks) < len(cols) and (j + k) < len(lines):
                        toks.extend(shlex.split(lines[j + k].strip()))
                        k += 1
                    if len(toks) >= max(idx_label, idx_auth, idx_chain) + 1:
                        try:
                            lab = int(toks[idx_label])
                            aut = int(toks[idx_auth])
                            ch  = toks[idx_chain]
                            rows.append((ch, lab, aut))
                        except Exception:
                            pass
                    j += k
        return rows
    except Exception:
        return []

def _parse_pdb_atoms(text: str):
    """PDB simple: returns list (chain, label_order, resSeq)."""
    rows = []
    for ln in text.splitlines():
        if ln.startswith("ATOM"):
            chain = ln[21:22].strip() or "A"
            try:
                resi = int(ln[22:26])
            except ValueError:
                continue
            rows.append((chain, resi))
    # build order 1..N per chain
    from collections import OrderedDict
    out = []
    per_chain = {}
    for ch, resi in rows:
        per_chain.setdefault(ch, OrderedDict())
        per_chain[ch][resi] = None
    for ch, od in per_chain.items():
        order = 1
        for resi in od.keys():
            out.append((ch, order, resi))
            order += 1
    return out

def _build_label_to_auth_maps(path: Path):
    """Return dict: chain -> {label_seq_id -> auth_seq_id}; for PDB uses order 1..N -> resSeq."""
    try:
        text = path.read_text(errors="ignore")
    except Exception:
        return {}
    try:
        if path.suffix.lower() in (".cif", ".mmcif"):
            rows = _parse_cif_atom_site(text)
        else:
            rows = _parse_pdb_atoms(text)
        from collections import OrderedDict
        chains = {}
        if path.suffix.lower() in (".cif", ".mmcif"):
            for ch, lab, aut in rows:
                d = chains.setdefault(ch, OrderedDict())
                if lab not in d:
                    d[lab] = aut
        else:
            for ch, lab, aut in rows:
                d = chains.setdefault(ch, OrderedDict())
                if lab not in d:
                    d[lab] = aut
        return {ch: dict(od) for ch, od in chains.items()}
    except Exception:
        return {}

def _color_with_label_to_auth(view, path: Path, highlight_positions: list[int], base_color="#87aade"):
    """Robust coloring: try label->auth mapping; if it fails, fall back to simple 'resi' selection."""
    # base style always
    view.setStyle({"cartoon": {"color": base_color}})
    try:
        maps = _build_label_to_auth_maps(path)  # chain -> {label: auth}
        if not maps:
            # Fallback simple coloring (may be off if numbering differs)
            sel = _compact_ranges(highlight_positions)
            if sel:
                view.setStyle({"resi": sel}, {"cartoon": {"color": "purple"}})
            return

        # choose the longest chain (likely main chain)
        chain_lengths = {ch: len(m) for ch, m in maps.items()}
        if not chain_lengths:
            return
        main_chain = max(chain_lengths, key=chain_lengths.get)
        m = maps[main_chain]

        auth_hits = sorted({m[p] for p in highlight_positions if p in m})
        sel = _compact_ranges(auth_hits)
        if sel:
            view.setStyle({"chain": main_chain, "resi": sel}, {"cartoon": {"color": "purple"}})

        missing = [p for p in highlight_positions if p not in m]
        if missing:
            st.caption(f"Note: {len(missing)} residues are not present in coordinates (disordered/trimmed).")
    except Exception as e:
        # Never block visualization
        st.warning(f"Epitope coloring fell back to simple mode: {e}")
        sel = _compact_ranges(highlight_positions)
        if sel:
            view.setStyle({"resi": sel}, {"cartoon": {"color": "purple"}})

# ---- CSV loader (supports span / residue schemas) ----------------------------
def _normalize_headers(cols: list[str]) -> list[str]:
    out = []
    for c in cols:
        c2 = str(c).strip().lower().replace(" ", "").replace("-", "").replace("_", "")
        c2 = (c2.replace("ó","o").replace("á","a").replace("é","e").replace("í","i").replace("ú","u"))
        out.append(c2)
    return out

def _pick_column(norm_cols: list[str], candidates: set[str]) -> int | None:
    for i, nc in enumerate(norm_cols):
        if nc in candidates:
            return i
    for i, nc in enumerate(norm_cols):
        if any(nc.startswith(c) or nc.endswith(c) for c in candidates):
            return i
    return None

def load_epitopes_csv(csv_path: Path) -> tuple[str, pd.DataFrame]:
    if not csv_path.exists():
        st.error(f"CSV not found: `{csv_path}`"); st.stop()
    try:
        df = pd.read_csv(csv_path, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(csv_path)

    orig_cols = list(df.columns.astype(str))
    norm_cols = _normalize_headers(orig_cols)

    prot_span = {"protein","name","antigen","sequencename","seqname","proteina"}
    start_syn = {"start","epitopestart","inicio","startpos","from"}
    end_syn   = {"end","epitopeend","fin","endpos","to"}
    prob_syn  = {"epitopeprobability","bepipredscore","score","prob","probability","bepipred2score"}

    prot_res  = {"entry","protein","name","antigen","sequencename","seqname"}
    pos_syn   = {"position","pos","residue","resid","resindex"}

    # A) span schema
    i_prot = _pick_column(norm_cols, prot_span)
    i_start = _pick_column(norm_cols, start_syn)
    i_end   = _pick_column(norm_cols, end_syn)
    i_prob  = _pick_column(norm_cols, prob_syn)

    if all(x is not None for x in (i_prot, i_start, i_end, i_prob)):
        mapping = {
            orig_cols[i_prot]:  "Protein",
            orig_cols[i_start]: "Start",
            orig_cols[i_end]:   "End",
            orig_cols[i_prob]:  "EpitopeProbability",
        }
        span_df = df.rename(columns=mapping)[["Protein","Start","End","EpitopeProbability"]].copy()
        span_df["Start"] = pd.to_numeric(span_df["Start"], errors="coerce")
        span_df["End"] = pd.to_numeric(span_df["End"], errors="coerce")
        span_df["EpitopeProbability"] = pd.to_numeric(span_df["EpitopeProbability"], errors="coerce")
        span_df = span_df.dropna(subset=["Protein","Start","End","EpitopeProbability"])
        swap = span_df["End"] < span_df["Start"]
        if swap.any():
            span_df.loc[swap, ["Start","End"]] = span_df.loc[swap, ["End","Start"]].values
        return "span", span_df

    # B) residue schema
    i_entry = _pick_column(norm_cols, prot_res)
    i_pos   = _pick_column(norm_cols, pos_syn)
    i_prob2 = _pick_column(norm_cols, prob_syn)

    if all(x is not None for x in (i_entry, i_pos, i_prob2)):
        mapping = {
            orig_cols[i_entry]: "Protein",
            orig_cols[i_pos]:   "pos",
            orig_cols[i_prob2]: "EpitopeProbability",
        }
        res_df = df.rename(columns=mapping)[["Protein","pos","EpitopeProbability"]].copy()
        res_df["pos"] = pd.to_numeric(res_df["pos"], errors="coerce")
        res_df["EpitopeProbability"] = pd.to_numeric(res_df["EpitopeProbability"], errors="coerce")
        res_df = res_df.dropna(subset=["Protein","pos","EpitopeProbability"])
        return "residue", res_df

    st.error(
        "Could not detect required columns in epitopes CSV.\n\n"
        "Accepted span schema: Protein/Name/Antigen + Start + End + EpitopeProbability\n"
        "Accepted residue schema: Entry/Protein/Name + Position + EpitopeProbability"
    )
    st.stop()

# ---- Position selection helpers ---------------------------------------------
def positions_above_threshold(schema: str, df: pd.DataFrame, protein: str, thr: float=0.5) -> list[int]:
    if schema == "span":
        sub = df[df["Protein"].astype(str).str.fullmatch(protein, case=False, na=False)]
        if sub.empty:
            tok = _norm_token(protein)
            sub = df[_norm_series_tokens(df["Protein"]).str.contains(tok)]
        sub = sub[sub["EpitopeProbability"] > thr]
        pos = []
        for _, r in sub.iterrows():
            s, e = int(r["Start"]), int(r["End"])
            pos.extend(range(min(s, e), max(s, e) + 1))
        return sorted(set(pos))

    # residue schema
    sub = df[df["Protein"].astype(str).str.fullmatch(protein, case=False, na=False)]
    if sub.empty:
        tok = _norm_token(protein)
        sub = df[_norm_series_tokens(df["Protein"]).str.contains(tok)]
    sub = sub[sub["EpitopeProbability"] > thr]
    return sorted(set(int(p) for p in sub["pos"].tolist()))

def per_residue_profile(schema: str, df: pd.DataFrame, protein: str) -> pd.DataFrame:
    """Return DataFrame with columns: pos, EpitopeProbability for selected protein."""
    if schema == "residue":
        sub = df[df["Protein"].astype(str).str.fullmatch(protein, case=False, na=False)]
        if sub.empty:
            tok = _norm_token(protein)
            sub = df[_norm_series_tokens(df["Protein"]).str.contains(tok)]
        prof = sub[["pos","EpitopeProbability"]].dropna().copy()
        prof["pos"] = prof["pos"].astype(int)
        return prof.sort_values("pos")

    # expand span schema to per-residue profile
    sub = df[df["Protein"].astype(str).str.fullmatch(protein, case=False, na=False)]
    if sub.empty:
        tok = _norm_token(protein)
        sub = df[_norm_series_tokens(df["Protein"]).str.contains(tok)]
    if sub.empty:
        return pd.DataFrame(columns=["pos","EpitopeProbability"])
    max_pos = int(sub[["Start","End"]].max().max())
    vals = np.full(max_pos + 1, np.nan)
    for _, r in sub.iterrows():
        s, e, sc = int(r["Start"]), int(r["End"]), float(r["EpitopeProbability"])
        s, e = min(s, e), max(s, e)
        current = vals[s:e+1]
        if np.isnan(current).all():
            vals[s:e+1] = sc
        else:
            vals[s:e+1] = np.nanmax(np.vstack([current, np.full_like(current, sc)]), axis=0)
    pos = np.arange(1, max_pos + 1)
    return pd.DataFrame({"pos": pos, "EpitopeProbability": vals[1:]})

# ---- Robust 3D renderer ------------------------------------------------------
def render_structure(protein: str, highlight_positions: list[int], *, base_color="#87aade"):
    """
    Robust 3D viewer:
    - Loads mmCIF/PDB
    - Tries label→auth mapping for coloring; if anything fails, falls back to simple 'resi' selection
    """
    path = find_structure_for(protein)
    if not path or not path.exists():
        st.warning(f"No structure file was found for **{protein}** in `{STRUCT_DIR}`.")
        return

    try:
        model_str = path.read_text(errors="ignore")
    except Exception as e:
        st.error(f"Could not read structure file: {e}")
        return

    fmt = "mmCIF" if path.suffix.lower() in (".cif", ".mmcif") else "pdb"

    view = py3Dmol.view(width=900, height=600)
    view.setBackgroundColor("white")
    try:
        view.addModel(model_str, fmt)
    except Exception:
        # Some builds accept 'cif' instead of 'mmCIF'
        alt_fmt = "cif" if fmt == "mmCIF" else fmt
        view.addModel(model_str, alt_fmt)

    # robust coloring with mapping (fallback on error)
    _color_with_label_to_auth(view, path, highlight_positions, base_color=base_color)

    view.zoomTo()
    st.caption(f"Using structure file: `{path.relative_to(STRUCT_DIR)}`")
    components.html(view._make_html(), height=620, scrolling=False)

# ==============================================================================
# UI
# ==============================================================================
st.title("BepiPred-2.0 — Epitope Coloring from CSV")

# Load CSV (supports residue/ span schemas)
schema, epi_df = load_epitopes_csv(CSV_PATH)

# Protein selector
proteins = sorted(epi_df["Protein"].dropna().astype(str).unique(), key=str.lower)
protein = st.selectbox("Protein", proteins, index=0)

THRESHOLD = 0.5

# ---- 1) DATA TABLE -----------------------------------------------------------
st.subheader("Data table")
if schema == "residue":
    tbl = epi_df[epi_df["Protein"].astype(str).str.fullmatch(protein, case=False, na=False)]
    if tbl.empty:
        tok = _norm_token(protein)
        tbl = epi_df[_norm_series_tokens(epi_df["Protein"]).str.contains(tok)]
    st.dataframe(
        tbl.sort_values("pos")[["Protein","pos","EpitopeProbability"]],
        use_container_width=True, hide_index=True
    )
else:  # span
    tbl = epi_df[epi_df["Protein"].astype(str).str.fullmatch(protein, case=False, na=False)]
    if tbl.empty:
        tok = _norm_token(protein)
        tbl = epi_df[_norm_series_tokens(epi_df["Protein"]).str.contains(tok)]
    st.dataframe(
        tbl.sort_values(["Start","End"])[["Protein","Start","End","EpitopeProbability"]],
        use_container_width=True, hide_index=True
    )

# ---- 2) CHART: EpitopeProbability vs Position --------------------------------
st.subheader("EpitopeProbability vs Position")
profile_df = per_residue_profile(schema, epi_df, protein).dropna()
profile_df["Above"] = profile_df["EpitopeProbability"] > THRESHOLD

chart_line = alt.Chart(profile_df).mark_line().encode(
    x=alt.X("pos:Q", title="Position"),
    y=alt.Y("EpitopeProbability:Q", title="EpitopeProbability", scale=alt.Scale(domain=[0, 1])),
    tooltip=[alt.Tooltip("pos:Q", title="Pos"),
             alt.Tooltip("EpitopeProbability:Q", title="Score", format=".3f")],
).properties(height=300)

chart_pts = alt.Chart(profile_df).mark_circle(size=36).encode(
    x="pos:Q",
    y="EpitopeProbability:Q",
    color=alt.condition("datum.Above", alt.value("purple"), alt.value("#3465a4")),
    tooltip=[alt.Tooltip("pos:Q", title="Pos"),
             alt.Tooltip("EpitopeProbability:Q", title="Score", format=".3f")],
)

rule = alt.Chart(pd.DataFrame({"y":[THRESHOLD]})).mark_rule(color="#999", strokeDash=[6,3]).encode(y="y:Q")

st.altair_chart((chart_line + chart_pts + rule).interactive(), use_container_width=True)

# ---- 3) 3D STRUCTURE ---------------------------------------------------------
st.subheader("Interactive 3D structure (cartoon, purple = > 0.5)")
highlight = positions_above_threshold(schema, epi_df, protein, thr=THRESHOLD)
render_structure(protein, highlight, base_color="#87aade")