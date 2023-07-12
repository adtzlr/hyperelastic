# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

# import os
# import hyperelastic

project = "Hyperelastic"
copyright = "2023, Andreas Dutzler"
author = "Andreas Dutzler"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "myst_nb",
]
source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
    ".myst": "myst-nb",
}
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
]
myst_url_schemes = ("http", "https", "mailto")

nb_execution_mode = "off"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

napoleon_google_docstring = False
napoleon_numpy_docstring = True
math_number_all = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_show_sourcelink = False
html_logo = "_static/logo.png"

# Define the json_url for our version switcher.
# json_url = "https://hyperelastic.readthedocs.io/en/latest/_static/switcher.json"

# Define the version we use for matching in the version switcher.
# version_match = os.environ.get("READTHEDOCS_VERSION")
# # If READTHEDOCS_VERSION doesn't exist, we're not on RTD
# # If it is an integer, we're in a PR build and the version isn't correct.
# if not version_match or version_match.isdigit():
#     # For local development, infer the version to match from the package.
#     release = hyperelastic.__version__
#     if "dev" in release or "rc" in release:
#         version_match = "latest"
#         # We want to keep the relative reference if we are in dev mode
#         # but we want the whole url if we are effectively in a released version
#         json_url = "_static/switcher.json"
#     else:
#         version_match = "v" + release

html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/adtzlr/hyperelastic",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
        {
            "name": "Read the Docs",
            "url": "https://readthedocs.org/projects/hyperelastic/downloads",
            "icon": "fa-solid fa-book",
            "type": "fontawesome",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/hyperelastic/",
            "icon": "fa-solid fa-box",
            "type": "fontawesome",
        }
    ],
    "logo": {
        #"image_light": "_static/logo-light.png",
        #"image_dark": "_static/logo-dark.png",
        "text": "Hyperelastic",
    },
    # "switcher": {
    #     "json_url": json_url,
    #     "version_match": version_match,
    # },
    # "navbar_start": ["navbar-logo", "version-switcher"],
}
