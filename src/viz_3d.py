import py3Dmol
from typing import List, Tuple, Optional, Dict

def _add_model(viewer: py3Dmol.view, struct_text: str, ext: str):
    if ext.lower().endswith(".pdb"):
        viewer.addModel(struct_text, "pdb")
    else:
        viewer.addModel(struct_text, "mmcif")

def render_structure(
    struct_path: str,
    background: str = "white",
    cartoon_opacity: float = 1.0,
    cartoon_thickness: float = 0.5,
    cartoon_arrows: bool = True,
    # modos de color
    color_mode: str = "by_chain", # 'by_chain' | 'uniform' | 'secondary'
    uniform_color: Optional[str] = "#1f77b4",
    chain_colors: Optional[Dict[str, str]] = None,
    # regiones por residuo
    region_colors: Optional[List[Tuple[Optional[str], int, int, str]]] = None,
    # override de α/β
    enable_secondary: bool = False,
    helix_color: str = "#1F4E79", # azul oscuro
    sheet_color: str = "#56B4E9", # cian
    # extras
    show_surface: bool = False,
    surface_opacity: float = 0.2,
    spin: bool = False,
    width: int = 960,
    height: int = 700
):
    with open(struct_path, "r") as f:
        text = f.read()
    ext = struct_path.lower()

    viewer = py3Dmol.view(width=width, height=height)
    _add_model(viewer, text, ext)

    # 1. Aplicar siempre el estilo de cinta (cartoon) como base
    viewer.setStyle(
        {},
        {"cartoon": {
            "opacity": float(cartoon_opacity),
            "thickness": float(cartoon_thickness),
            "arrows": bool(cartoon_arrows)
        }}
    )

    # 2. Aplicar el modo de color principal sobre el estilo base
    if color_mode == "uniform" and uniform_color:
        viewer.setStyle({"cartoon": True}, {"color": uniform_color})
    elif color_mode == "by_chain" and chain_colors:
        for ch, col in chain_colors.items():
            viewer.setStyle({"chain": ch}, {"cartoon": {"color": col}})
    
    # 3. Aplicar colores de región sobre la coloración principal
    if region_colors:
        for chain, a, b, col in region_colors:
            sel = {"resi": list(range(a, b + 1))}
            if chain:
                sel["chain"] = chain
            viewer.setStyle(sel, {"cartoon": {"color": col}})
            
    # 4. Aplicar override de estructura secundaria
    if enable_secondary and color_mode == "secondary":
        viewer.setStyle({"ss": "h"}, {"cartoon": {"color": helix_color}})
        viewer.setStyle({"ss": "s"}, {"cartoon": {"color": sheet_color}})

    # Extras
    viewer.setBackgroundColor(background)
    if show_surface:
        viewer.addSurface(py3Dmol.VDW, {"opacity": float(surface_opacity)})
    viewer.zoomTo()
    if spin:
        viewer.spin(True)
        
    return viewer