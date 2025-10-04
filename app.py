import streamlit as st

st.set_page_config(page_title="Structural and Immunogenic Evaluation of Silk Proteins Using Advanced Bioinformatics and Deep Learning for Biomaterials Applications", layout="wide")
st.title("Structural and Immunogenic Evaluation of Silk Proteins Using Advanced Bioinformatics and Deep Learning for Biomaterials Applications")
st.write("**Authors names**:  Puerta-González Andrés, Soto-Ospina Alejandro, Montoya Osorio Yuliet, Bustamante-Osorno John, Salazar-Peláez Lina María")

st.markdown("This Streamlit application provides an interactive platform to visualize and analyze the key findings from a research article on silk proteins, focusing on their use in biomaterials. The app is divided into distinct sections, allowing users to explore the structural, immunological, and phylogenetic data in detail.")

st.header("About the Project")
st.markdown("""
This project leverages advanced bioinformatics and deep learning techniques to evaluate the structural and immunogenic properties of silk proteins, specifically **Fibroin light chain (FibL)**, **Fibroin heavy chain (FibH)**, **P25**, **Sericin1 - 4 proteins**.

Use the navigation bar on the left to explore the different sections of the analysis:
* **3D Structures**: Interactive visualization of protein structures.
* **Immunogenicity**: Analysis of predicted B-cell and T-cell epitopes.
* **Results and Analysis**: Detailed breakdown of the key findings, including epitope predictions.
* **Data and Metrics**: Access to raw data tables and figures.
* **Files and Downloads**: A section to download relevant files for further use.
""")

st.info("Navigate through the sections using the sidebar to the left. The sidebar will appear automatically on desktop, or you can expand it on mobile.")