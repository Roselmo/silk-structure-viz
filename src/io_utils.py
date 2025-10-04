from pathlib import Path
import pandas as pd
import streamlit as st
import requests

DATA_DIR = Path("data")

@st.cache_data
def list_structure_files():
    sdir = DATA_DIR / "structures"
    return sorted([*sdir.glob("*.pdb"), *sdir.glob("*.cif")])

@st.cache_data
def load_table(name: str) -> pd.DataFrame:
    p = DATA_DIR / "tables" / name
    if not p.is_file():
        raise FileNotFoundError(f"Table file not found: {p}")
        
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    if p.suffix.lower() == ".tsv":
        return pd.read_csv(p, sep="\t")
    if p.suffix.lower() in [".xls", ".xlsx"]:
        return pd.read_excel(p)
    if p.suffix.lower() == ".json":
        return pd.read_json(p)
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
        
    raise ValueError(f"Unsupported table format: {p.suffix}")

@st.cache_data
def fetch_structure_from_url(url: str) -> str:
    """Download a PDB/mmCIF file to a temporary cache and return the path."""
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    # Derive filename from URL
    name = url.split("/")[-1] or "structure.cif"
    dest = DATA_DIR / "structures" / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        f.write(r.content)
    return str(dest)