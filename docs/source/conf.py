from sarkit import _version

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "sarkit"
copyright = "2024-%Y, Valkyrie Systems Corporation"
author = "Valkyrie Systems Corporation"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.duration",
    "sphinx.ext.intersphinx",
    "numpydoc",
    "sphinx_rtd_theme",
    "sphinxcontrib.autoprogram",
]

templates_path = ["_templates"]
exclude_patterns = []
default_role = "any"

version = _version.__version__

suppress_warnings = [
    "ref.citation",
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_logo = "_static/sarkit_logo.png"

html_theme_options = {
    "navigation_depth": -1,
}

autosummary_generate = True
autosummary_generate_overwrite = False
autodoc_typehints = "none"
add_module_names = False

# doctest
doctest_test_doctest_blocks = ""  # don't test unmarked blocks
doctest_global_setup = """
import pathlib
import tempfile
tmpdir = tempfile.TemporaryDirectory()
tmppath = pathlib.Path(tmpdir.name)
"""
doctest_global_cleanup = """
tmpdir.cleanup()
"""
doctest_show_successes = False

# intersphinx
intersphinx_mapping = {
    "lxml": ("https://lxml.de/apidoc/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "python": ("https://docs.python.org/3", None),
}

# numpydoc
numpydoc_xref_param_type = True
numpydoc_xref_ignore = {"optional", "of", "N"}
numpydoc_validation_checks = {
    "PR10",  # requires a space before the colon separating the parameter name and type
    "GL07",  # Sections are in the wrong order
}
numpydoc_show_inherited_class_members = {
    "sarkit.cphd.ElementWrapper": False,
    "sarkit.crsd.ElementWrapper": False,
    "sarkit.sicd.ElementWrapper": False,
    "sarkit.sidd.ElementWrapper": False,
    "sarkit.xmlhelp.ElementWrapper": False,
}
