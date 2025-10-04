import hashlib
import streamlit as st
import streamlit.components.v1 as components
from src.io_utils import list_structure_files
from src.viz_3d import render_structure
from src.color_utils import parse_ranges

st.set_page_config(
    page_title="3D Structures",
    layout="wide"
)

st.title("Interactive 3D Structures")
st.markdown("Explore the three-dimensional structures of the silk proteins: Fibroin light chain (FILC), Fibroin heavy chain (FIHC), and P25. Use the controls to change the rendering style, color schemes, and highlight specific regions.")

# Only local structures
files = list_structure_files()
if not files:
    st.warning("No structures found in `data/structures/`. Add .pdb/.cif files and reload.")
    st.stop()

names = [p.name for p in files]
pick = st.selectbox("Select a structure", names, index=0)
struct_path = str([p for p in files if p.name == pick][0])

st.divider()
st.subheader("Ribbon controls")
c1, c2, c3 = st.columns(3)

with c1:
    background = st.color_picker("Background", value="#FFFFFF")
    cartoon_opacity = st.slider("Ribbon opacity", 0.1, 1.0, 1.0, 0.05)
with c2:
    cartoon_thickness = st.slider("Ribbon thickness", 0.1, 1.0, 0.5, 0.05)
    cartoon_arrows = st.checkbox("Show arrows on β-sheets", value=True)
with c3:
    spin = st.checkbox("Spin", value=False)
    show_surface = st.checkbox("Surface (VDW)", value=False)
    surface_opacity = st.slider("Surface opacity", 0.0, 1.0, 0.2, 0.05)

st.divider()
st.subheader("Color modes")
colA, colB = st.columns([1, 2])
with colA:
    color_mode = st.selectbox(
        "Select color mode",
        ["by_chain (recommended for complexes)", "uniform", "secondary (α/β)"],
        index=0
    )
    
# Mapea el texto del selectbox a un modo de color interno
mode_map = {
    "by_chain (recommended for complexes)": "by_chain",
    "uniform": "uniform",
    "secondary (α/β)": "secondary",
}
selected_mode = mode_map[color_mode]

chain_colors = {}
uniform_color = "#1f77b4"
helix_color, sheet_color = "#1F4E79", "#56B4E9"

with colB:
    if selected_mode == "by_chain":
        st.caption("Provide chain IDs and colors (e.g., A,B or A,B,C). Useful for dimers/complexes.")
        chain_list = st.text_input("Chains (comma-separated)", value="A,B")
        for ch in [x.strip() for x in chain_list.split(",") if x.strip()]:
            chain_colors[ch] = st.color_picker(f"Color for chain {ch}", value="#56B4E9", key=f"col_{ch}")
    elif selected_mode == "uniform":
        uniform_color = st.color_picker("Uniform ribbon color", value="#1f77b4")
    elif selected_mode == "secondary":
        st.caption("Set α-helix and β-sheet colors (override).")
        hcol, scol = st.columns(2)
        with hcol:
            helix_color = st.color_picker("α-helix color", value="#1F4E79", key="helix_col")
        with scol:
            sheet_color = st.color_picker("β-sheet color", value="#56B4E9", key="sheet_col")

# Region coloring by residue indices (with optional chain)
st.divider()
st.subheader("Region coloring (by residue indices)")
st.caption("Define residue ranges like `1-50, 120-140` (optionally per chain).")
region_entries = []
enable_regions = st.checkbox("Enable region coloring", value=False)
if enable_regions:
    n_regs = st.number_input("Number of regions", min_value=1, max_value=20, value=1, step=1)
    for i in range(n_regs):
        cc1, cc2, cc3, _ = st.columns([1, 2, 2, 1])
        chain = cc1.text_input(f"Chain {i+1} (optional)", value="", key=f"reg_chain_{i}")
        ranges = cc2.text_input(f"Residue ranges {i+1}", value="", key=f"reg_ranges_{i}")
        col = cc3.color_picker("Color", value="#F0E442", key=f"reg_color_{i}")
        if ranges:
            for a, b in parse_ranges(ranges):
                region_entries.append((chain or None, a, b, col))

# Render viewer object
viewer = render_structure(
    struct_path=struct_path,
    background=background,
    cartoon_opacity=float(cartoon_opacity),
    cartoon_thickness=float(cartoon_thickness),
    cartoon_arrows=bool(cartoon_arrows),
    color_mode=selected_mode,
    uniform_color=uniform_color,
    chain_colors=chain_colors if selected_mode == "by_chain" else None,
    region_colors=region_entries if enable_regions else None,
    enable_secondary=(selected_mode == "secondary"),
    helix_color=helix_color,
    sheet_color=sheet_color,
    show_surface=show_surface,
    surface_opacity=float(surface_opacity),
    spin=spin,
    width=960,
    height=700
)

# ----- Force refresh WITHOUT using 'key' in components.html -----
# Build viewer HTML
html = viewer._make_html()

# Build a state hash that changes whenever any control/state changes
state_str = "|".join([
    struct_path, background,
    f"{cartoon_opacity:.2f}", f"{cartoon_thickness:.2f}", str(cartoon_arrows),
    selected_mode, uniform_color,
    ",".join([f"{k}:{v}" for k, v in chain_colors.items()]),
    ",".join([f"{r[0]}:{r[1]}-{r[2]}:{r[3]}" for r in region_entries]),
    helix_color, sheet_color,
    str(show_surface), f"{surface_opacity:.2f}", str(spin)
])
state_hash = hashlib.md5(state_str.encode()).hexdigest()

# Append the hash as an HTML comment so the HTML changes and Streamlit re-renders
html += f""

# Render iframe (no 'key' argument)
components.html(html, height=720, scrolling=False)

# Download button
with open(struct_path, "rb") as fh:
    st.download_button("Download structure", data=fh.read(), file_name=pick)