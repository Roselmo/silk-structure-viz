# pages/1_🧬_3D_Structures.py
from __future__ import annotations

from pathlib import Path
import re
from typing import List, Tuple, Optional

import streamlit as st
import streamlit.components.v1 as components

# Visor 3D
try:
    import py3Dmol
except Exception:
    py3Dmol = None

# ===============================
# 🔧 PAGE CONFIGURATION
# ===============================
st.set_page_config(
    page_title="3D Structural Prediction and Protein Interactions (AlphaFold3)",
    layout="wide"
)

st.title("3D Structural Prediction and Protein Interactions using AlphaFold3")
st.write("**Silk Protein Structural Modeling and Interaction Analysis**")

# ===============================
# 🧠 INTRODUCTION (concise & formal)
# ===============================
st.markdown("""
The three-dimensional structures of **FibL (Fibroin Light Chain)**, **FibH (Fibroin Heavy Chain)**, **P25**, and **Sericins 1–4** were predicted with **AlphaFold3** (AF3), the latest model from *Google DeepMind* and *Isomorphic Labs*. AF3 introduces a **diffusion-based** framework that extends beyond single-chain folding to **multi-chain complexes** and diverse biomolecular interactions. This enables the exploration of silk-protein organization and interfaces that underpin mechanical performance and biocompatibility in silk-based biomaterials.

**Complexes evaluated in this work:**
- 6 × FibL + 1 × P25  
- 6 × FibL + 1 × P25 + 1 × Sericin2  
- 6 × FibL + 1 × P25 + 1 × Sericin2 + 1 × Sericin4  
- 6 × FibL + 1 × P25 + 1 × Sericin3  
- 6 × FibL + 1 × P25 + 1 × Sericin4  
- 1 × FibL + 1 × FibH + 1 × Sericin4

**Why AF3 matters here.** Compared with AF2, AF3 improves complex assembly modeling and protein–protein interfaces—useful to hypothesize how **FibL, FibH, P25,** and **Sericins** arrange and contact. As with all prediction tools, AF3 still struggles with **intrinsically disordered regions**, alternative **conformations**, and especially **RNA** systems; therefore, models should be treated as **testable hypotheses** and refined with **experimental data** or **molecular simulations**.

*Reference:* DeepMind & Isomorphic Labs. *AlphaFold3: A Unified Model for Biomolecular Structure and Interaction Prediction.* (May 2024).
""")

# ===============================
# 📁 Paths & helpers
# ===============================
STRUCT_DIR = Path("data") / "structures"  # where your .cif/.pdb live

def norm_name(s: str) -> str:
    """Normalize common silk names to canonical short labels."""
    s = s.strip()
    s2 = re.sub(r"[\s_]+", "", s, flags=re.IGNORECASE).lower()
    if s2 in ("filc", "fibroinlightchain", "fibl", "lightchain", "lchain"):
        return "FibL"
    if s2 in ("fihc", "fibroinheavychain", "fibh", "heavychain", "hchain"):
        return "FibH"
    if s2 in ("p25", "p25protein"):
        return "P25"
    for k in range(1, 4 + 1):
        if s2 in (f"sericin{k}", f"sericina{k}"):
            return f"Sericin{k}"
    if s in {"FibL", "FibH", "P25"} or s.startswith("Sericin"):
        return s
    return s

def find_structure_for(name: str) -> Optional[Path]:
    """
    Find a local .cif/.pdb for a given protein/complex name using flexible patterns.
    Avoids '**' (pathlib restriction) and supports 'Sericin'/'Sericina'.
    """
    if not STRUCT_DIR.exists():
        return None

    n = norm_name(name)
    patterns = [
        f"{n}*.cif", f"{n}_*.cif", f"*{n}*.cif",
        f"{n}*.pdb", f"{n}_*.pdb", f"*{n}*.pdb",
    ]
    if n.startswith("Sericin"):
        k = n.replace("Sericin", "")
        patterns += [f"Sericina{k}*.cif", f"*Sericina{k}*.cif",
                     f"Sericina{k}*.pdb", f"*Sericina{k}*.pdb"]

    for pat in patterns:
        hits = sorted(STRUCT_DIR.glob(pat))
        if hits:
            return hits[0]
    return None

def list_all_structures() -> List[Path]:
    if not STRUCT_DIR.exists():
        return []
    return sorted(list(STRUCT_DIR.glob("*.cif")) + list(STRUCT_DIR.glob("*.pdb")))

def parse_ranges(ranges_str: str) -> List[Tuple[int, int]]:
    """Parse '10-25, 120-130, 205' → [(10,25), (120,130), (205,205)]"""
    spans: List[Tuple[int, int]] = []
    if not ranges_str:
        return spans
    for p in ranges_str.split(","):
        s = p.strip()
        m = re.match(r"^(\d+)\s*-\s*(\d+)$", s)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            spans.append((min(a, b), max(a, b)))
        elif s.isdigit():
            i = int(s)
            spans.append((i, i))
    return spans

# ===============================
# 🎨 Cartoon styles (ribbon-like helices)
# ===============================
def set_cartoon_base(view, color="#93a7cf"):
    # Cartoon ribbon without cylinders or arrows
    view.setStyle({}, {"cartoon": {"color": color}})

def color_by_secondary_structure(view, helix="#e76f51", sheet="#2a9d8f"):
    # Helix/Sheet cartoon, ribbon-like
    view.setStyle({"ss": "h"}, {"cartoon": {"color": helix}})
    view.setStyle({"ss": "s"}, {"cartoon": {"color": sheet}})

def color_by_chain(view):
    palette = [
        "#2F4B7C", "#A05195", "#D45087", "#F95D6A", "#FF7C43",
        "#FFA600", "#003F5C", "#665191", "#BC5090", "#FF6361"
    ]
    chains = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
    for i, ch in enumerate(chains):
        view.setStyle({"chain": ch}, {"cartoon": {"color": palette[i % len(palette)]}})

def color_ranges(view, ranges: List[Tuple[int, int]], color="#6D2E8C"):
    for a, b in ranges:
        view.setStyle({"resi": list(range(a, b + 1))}, {"cartoon": {"color": color}})

def show_structure(cif_or_pdb: Path, mode: str, highlight: List[Tuple[int, int]]):
    if py3Dmol is None:
        st.error("py3Dmol is not available. Add `py3Dmol` to requirements.txt and reinstall.")
        return

    data = cif_or_pdb.read_text(errors="ignore")
    fmt = "mmCIF" if cif_or_pdb.suffix.lower() == ".cif" else "pdb"
    view = py3Dmol.view(width=1000, height=720)
    view.addModel(data, fmt)
    set_cartoon_base(view)

    if mode == "Cartoon (Helix/Sheet colors)":
        color_by_secondary_structure(view)
    elif mode == "By chain (cartoon)":
        color_by_chain(view)

    if highlight:
        color_ranges(view, highlight)

    view.zoomTo()
    view.setBackgroundColor("white")
    components.html(view._make_html(), height=740, scrolling=False)

# ===============================
# 🧭 UI: selector & viewer
# ===============================
st.subheader("Interactive 3D structures (AlphaFold3)")

tab1, tab2 = st.tabs(["Single protein", "Complex"])

with tab1:
    c1, c2 = st.columns([1, 1])
    with c1:
        protein = st.selectbox(
            "Select protein",
            ["FibL", "FibH", "P25", "Sericin1", "Sericin2", "Sericin3", "Sericin4"],
            index=0
        )
    with c2:
        style = st.selectbox(
            "Style",
            ["Cartoon (Helix/Sheet colors)", "By chain (cartoon)"],
            index=0
        )

    highlight_str = st.text_input(
        "Color specific residue ranges (e.g., 10-25, 120-130, 205)",
        value=""
    )
    highlight = parse_ranges(highlight_str)

    path = find_structure_for(protein)
    if path and path.exists():
        st.caption(f"Source file: `{path}`")
        show_structure(path, style, highlight)
        st.download_button("📥 Download structure file", data=path.read_bytes(),
                           file_name=path.name,
                           mime="chemical/x-mmcif" if path.suffix.lower()==".cif" else "chemical/x-pdb")
    else:
        st.warning(
            f"No structure file was found for **{protein}** in `{STRUCT_DIR}`.\n"
            f"Place a `.cif`/`.pdb` named like `{protein}.cif`, `{protein}_*.cif`, or `*{protein}*.cif`."
        )

with tab2:
    st.markdown("Select a **prebuilt complex** file (.cif/.pdb) placed under `data/structures/`.")
    files = list_all_structures()
    if files:
        names = [p.name for p in files]
        pick = st.selectbox("Choose complex file", names, index=0, key="complex_pick")
        style2 = st.selectbox("Style", ["Cartoon (Helix/Sheet colors)", "By chain (cartoon)"], index=0, key="complex_style")
        highlight_str2 = st.text_input("Color specific residue ranges (e.g., 10-25, 120-130, 205)", value="", key="complex_ranges")
        highlight2 = parse_ranges(highlight_str2)

        chosen = STRUCT_DIR / pick
        st.caption(f"Source file: `{chosen}`")
        show_structure(chosen, style2, highlight2)
        st.download_button("📥 Download complex file",
                           data=chosen.read_bytes(),
                           file_name=chosen.name,
                           mime="chemical/x-mmcif" if chosen.suffix.lower()==".cif" else "chemical/x-pdb")
    else:
        st.warning("No `.cif`/`.pdb` files found in `data/structures/`.")