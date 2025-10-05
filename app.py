# app.py
from __future__ import annotations
import base64
from pathlib import Path
import random
import string
import streamlit as st

# ===============================
# 🔧 PAGE CONFIGURATION
# ===============================
st.set_page_config(
    page_title="Structural and Immunogenic Evaluation of Silk Proteins Using Advanced Bioinformatics and Deep Learning for Biomaterials Applications",
    layout="wide"
)

st.title("Structural and Immunogenic Evaluation of Silk Proteins Using Advanced Bioinformatics and Deep Learning for Biomaterials Applications")
st.write("**Authors names**:  Puerta-González Andrés, Soto-Ospina Alejandro, Montoya Osorio Yuliet, Bustamante-Osorno John, Salazar-Peláez Lina María")

# ===== Sidebar branding above nav (ADD RIGHT AFTER st.set_page_config) =====
import base64
from pathlib import Path
import streamlit as st

# Ruta de tu imagen
_SILK_IMG = Path("data/assets/1200px-CSIRO_ScienceImage_10746_An_adult_silkworm_moth.jpg")

def _img_data_uri(p: Path) -> str:
    try:
        b = p.read_bytes()
        return "data:image/jpeg;base64," + base64.b64encode(b).decode("utf-8")
    except Exception:
        return ""

# CSS:
#  - Oculta el rótulo "app" (primer div del nav)
#  - Reserva espacio para el header (ajusta la altura si cambias el tamaño de imagen)
#  - Fija el header en la parte superior del sidebar
st.markdown(
    """
    <style>
    /* Hide default group label (often "app") */
    section[data-testid="stSidebarNav"] > div:first-child {
        display: none !important;
    }
    /* Push the nav down so our brand fits above it.
       Ajusta esta altura si cambias el tamaño de la imagen/título. */
    section[data-testid="stSidebarNav"] {
        padding-top: 260px !important;
    }
    /* Sidebar brand container fixed at top */
    #sb-brand {
        position: absolute;
        top: 0; left: 0; right: 0;
        padding: 10px 12px 8px 12px;
        background: var(--secondary-background-color, #f7f7f7);
        border-bottom: 1px solid rgba(0,0,0,.06);
        z-index: 999;
    }
    #sb-brand .sb-img {
        width: 100%;
        border-radius: 8px;
        display: block;
    }
    #sb-brand h3 {
        margin: .55rem 0 0;
        font-weight: 800;
        font-size: 1.05rem;
        display: flex;
        align-items: center;
        gap: .4rem;
    }
    #sb-brand h3::before { content: "🧬"; }
    #sb-brand p {
        margin: .25rem 0 0;
        font-size: .85rem;
        opacity: .9;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Render HTML del header (antes del nav) usando data URI para la imagen
_data_uri = _img_data_uri(_SILK_IMG)
_brand_html = f"""
<div id="sb-brand">
  {'<img class="sb-img" src="'+_data_uri+'" alt="Silkworm (Bombyx mori)">' if _data_uri else ''}
  <h3>Silk Protein Explorer</h3>
  <p>Structural &amp; immunogenic analysis of silk proteins</p>
</div>
"""
# Colócalo en el sidebar (queda por encima del nav gracias al CSS anterior)
st.sidebar.markdown(_brand_html, unsafe_allow_html=True)
# ===== end add-on =====

# -----------------------------------------------------------------------------
# Helpers for robust hero banner
# -----------------------------------------------------------------------------
def pick_first_existing(candidates: list[Path]) -> Path | None:
    for p in candidates:
        if p.exists():
            return p
    return None

def to_base64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def render_hero_banner(caption: str = "", height_px: int = 320) -> None:
    """
    Render a CSS hero banner. Tries multiple candidate paths and, if found,
    embeds the image as base64 (data URI) to avoid path issues on Streamlit.
    """
    here = Path(__file__).parent.resolve()
    root = here

    candidates = [
        root / "data" / "assets" / "silk_hero.jpg",
        root / "data" / "assets" / "silk_hero.jpeg",
        root / "data" / "assets" / "silk-worm-destacada_opt.jpg",
        root / "data" / "assets" / "silk-worm-destacada_opt.jpeg",
        root / "data" / "silk_hero.jpg",
        root / "silk_hero.jpg",
    ]

    img_path = pick_first_existing(candidates)

    if img_path is None:
        st.warning(
            "⚠️ Hero image not found. I looked for:\n\n- " +
            "\n- ".join(str(p) for p in candidates)
        )
        return

    ext = img_path.suffix.lower().lstrip(".")
    if ext not in {"jpg", "jpeg", "png"}:
        ext = "jpg"
    b64 = to_base64(img_path)
    data_uri = f"data:image/{'jpeg' if ext in ('jpg','jpeg') else 'png'};base64,{b64}"

    suf = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))

    st.markdown(
        f"""
        <style>
        .hero-{suf} {{
            width: 100%;
            height: {height_px}px;
            background-image: url('{data_uri}');
            background-size: cover;
            background-position: center;
            border-radius: 12px;
            margin-bottom: 1.5rem;
            box-shadow: 0px 3px 15px rgba(0,0,0,0.3);
        }}
        .hero-caption-{suf} {{
            position: relative;
            top: {max(0, height_px - 100)}px;
            background: rgba(0,0,0,0.45);
            color: #fff;
            padding: 0.8rem 1.4rem;
            border-radius: 8px;
            font-weight: 600;
            font-size: 1.2rem;
            margin-left: 1rem;
            display: inline-block;
        }}
        </style>

        <div class="hero-{suf}">
            <div class="hero-caption-{suf}">
                {caption}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
# Hero banner (Option B)
# -----------------------------------------------------------------------------
render_hero_banner(
    caption="Bombyx mori cocoon & silkworm — project overview",
    height_px=320,
)

# -----------------------------------------------------------------------------
# Main content
# -----------------------------------------------------------------------------
st.markdown("""
### 🧬 Overview
This dashboard presents an integrated **structural and immunogenic evaluation** of *Bombyx mori* silk proteins, 
leveraging advanced bioinformatics and deep learning tools to support their use as potential biomaterials in regenerative medicine.
The analyses include epitope prediction, secondary structure mapping, and accessibility evaluation across multiple silk components.
""")

st.markdown("---")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Proteins analyzed", value="7", delta="FibH, FibL, P25, Sericin1–4")
with col2:
    st.metric(label="Epitope models", value="2", delta="BepiPred 2.0 & NetMHCpan")
with col3:
    st.metric(label="Predictions evaluated", value=">15,000", delta="per class I & II alleles")

st.markdown("""
### 🧫 Explore Modules
Use the navigation sidebar to explore each analysis section:

- **3D structures (AlphaFold3)** — Interactive cartoon-style models with residue/segment coloring and chain-level highlighting.  
- **BepiPred-2.0** — Sequential B-cell epitope prediction.  
- **T-cell Class I** — MHC-I binding affinity and antigen processing.  
- **T-cell Class II** — MHC-II elution and cleavage probability.  

Each page includes downloadable datasets and publication-ready visualizations.
""")

st.markdown("---")

st.markdown("""
### 🧵 Project context
This project integrates immunoinformatic approaches with molecular modeling to identify 
**non-immunogenic silk-derived proteins** suitable for biomedical applications such as 
**bioactive cardiac patches** and **biocompatible scaffolds**.
""")

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown(
    """
    <hr style='border:1px solid #ddd; margin-top:40px; margin-bottom:10px;'>

    <div style='text-align:center; color:gray; font-size:0.9rem;'>
        © 2025 Silk Immuno Project | Developed by Andrés Puerta González
    </div>
    """,
    unsafe_allow_html=True
)