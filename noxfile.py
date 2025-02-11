import os
import pathlib

import nox

os.environ.update({"PDM_IGNORE_SAVED_PYTHON": "1"})

nox.options.sessions = (
    "lint",
    "data",
    "test",
)


@nox.session
def docs(session):
    session.run_install("pdm", "sync", "-G", "all", "-G", "doc", external=True)
    session.run(
        "sphinx-build",
        "docs/source",
        "docs/build",
        "--fail-on-warning",
        "--keep-going",
        "--fresh-env",
        "--write-all",
        *session.posargs,
    )
    session.run(
        "sphinx-build",
        "-M",
        "doctest",
        "docs/source",
        "docs/build",
    )


@nox.session
def format(session):
    session.run_install("pdm", "sync", "--prod", "-G", "dev-lint", external=True)
    session.run("ruff", "check", "--fix")
    session.run("ruff", "format")


@nox.session
def lint(session):
    session.run_install("pdm", "sync", "--prod", "-G", "dev-lint", external=True)
    session.run("ruff", "check")
    session.run(
        "ruff",
        "format",
        "--diff",
    )
    session.run("mypy", pathlib.Path(__file__).parent / "sarkit")


@nox.session
def test(session):
    for next_session in ("test_standards", "test_processing", "test_verification"):
        session.notify(next_session)


@nox.session
def test_standards(session):
    session.run_install("pdm", "sync", "--prod", "-G", "dev-test", external=True)
    session.run("pytest", "tests/core", "tests/standards")


@nox.session
def test_processing(session):
    session.run_install(
        "pdm",
        "sync",
        "--prod",
        "-G",
        "dev-test",
        "-G",
        "processing",
        external=True,
    )
    session.run("pytest", "tests/processing")


@nox.session
def test_verification(session):
    session.run_install(
        "pdm",
        "sync",
        "--prod",
        "-G",
        "dev-test",
        "-G",
        "verification",
        external=True,
    )
    session.run("pytest", "tests/verification")


@nox.session
def data(session):
    session.run_install("pdm", "sync", "--prod", external=True)
    session.run(
        "python", "data/syntax_only/sicd/make_syntax_only_sicd_xmls.py", "--check"
    )
    session.run(
        "python", "data/syntax_only/cphd/make_syntax_only_cphd_xmls.py", "--check"
    )
