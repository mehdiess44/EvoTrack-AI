import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../src"))

project = "EvoTrack-AI"
author = "EvoTrack-AI"
copyright = f"{datetime.now().year}, {author}"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
]

autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"

autodoc_mock_imports = [
    "cv2",
    "faiss",
    "matplotlib",
    "nibabel",
    "numpy",
    "optuna",
    "pandas",
    "PIL",
    "pydicom",
    "psutil",
    "sentence_transformers",
    "seaborn",
    "SimpleITK",
    "skimage",
    "sklearn",
    "streamlit",
    "tensorflow",
    "torch",
    "transformers",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
language = "fr"

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_title = "Documentation EvoTrack-AI"

rst_epilog = """
.. |warning_medical| replace:: Prototype académique : EvoTrack-AI ne constitue pas un outil de diagnostic médical.
"""
