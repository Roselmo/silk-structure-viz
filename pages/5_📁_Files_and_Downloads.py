import streamlit as st
from pathlib import Path
import os

st.set_page_config(
    page_title="Downloads",
    layout="wide"
)

st.title("Files and Downloads")
st.markdown("Download the raw data, figures, and structures from the project.")

def get_files_in_dir(path):
    files = []
    if os.path.isdir(path):
        for f in os.listdir(path):
            full_path = os.path.join(path, f)
            if os.path.isfile(full_path):
                files.append(full_path)
    return files

st.header("Structures (.pdb / .cif)")
struct_path = Path("data") / "structures"
struct_files = get_files_in_dir(struct_path)
if struct_files:
    for f in struct_files:
        with open(f, "rb") as fp:
            st.download_button(
                label=f"Download {os.path.basename(f)}",
                data=fp,
                file_name=os.path.basename(f)
            )
else:
    st.info("No structure files found in `data/structures/`.")

st.header("Tables (.csv / .tsv)")
table_path = Path("data") / "tables"
table_files = get_files_in_dir(table_path)
if table_files:
    for f in table_files:
        with open(f, "rb") as fp:
            st.download_button(
                label=f"Download {os.path.basename(f)}",
                data=fp,
                file_name=os.path.basename(f)
            )
else:
    st.info("No table files found in `data/tables/`.")

st.header("Figures and Images")
image_path = Path("data") / "images"
image_files = get_files_in_dir(image_path)
if image_files:
    for f in image_files:
        with open(f, "rb") as fp:
            st.download_button(
                label=f"Download {os.path.basename(f)}",
                data=fp,
                file_name=os.path.basename(f)
            )
else:
    st.info("No image files found in `data/images/`.")