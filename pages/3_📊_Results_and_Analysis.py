import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="Results and Analysis",
    layout="wide"
)

st.title("Results and Analysis")
st.markdown("This section presents the key findings from the study, organized by the specific analyses performed.")

# --- Structural Analysis ---
st.header("1. Structural Analysis of Silk Proteins")
st.markdown("This subsection focuses on the three-dimensional structures of silk proteins, including **Fibroin light chain (FILC)**, **Fibroin heavy chain (FIHC)**, and **P25 protein**, and their domains. The visualization allows for the identification of their distinct characteristics, such as the $\\beta$-sheet dominance in FIHC and the unstructured nature of P25, which are crucial for understanding their mechanical and immunogenic properties.")

# --- In-silico Phylogenetics ---
st.header("2. In-silico Phylogenetics")
st.markdown("This subsection provides a phylogenetic analysis to understand the evolutionary relationships of the silk proteins. The results show that **P25 protein** is highly conserved across different Bombyx mori strains, while the **heavy and light chains of Fibroin** exhibit greater variability. This analysis helps to explain differences in protein structure and function across species.")