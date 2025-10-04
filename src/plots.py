import numpy as np
import matplotlib.pyplot as plt

def plot_pae_heatmap(pae_json: dict):
    """PAE: expected position error (n x n). Asume pae_json con key 'pae' o formato AF3."""
    # Ajusta este parser a tu formato (AF2/AF3 cambian keys; si tienes matriz directa, úsala).
    # Ejemplo genérico:
    if "pae" in pae_json:  # lista de dicts con i,j,pae
        # reconstruir matriz
        # Suponiendo pae_json["N"] = tamaño
        n = pae_json.get("N") or int(np.sqrt(len(pae_json["pae"])))
        M = np.zeros((n,n))
        for e in pae_json["pae"]:
            M[e["i"], e["j"]] = e["pae"]
        mat = M
    elif "predicted_aligned_error" in pae_json:
        mat = np.array(pae_json["predicted_aligned_error"])
    else:
        raise ValueError("Formato PAE no reconocido")

    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(mat, origin="lower")
    ax.set_title("Predicted Aligned Error (PAE)")
    ax.set_xlabel("Residue")
    ax.set_ylabel("Residue")
    fig.colorbar(im, ax=ax, label="Å")
    return fig