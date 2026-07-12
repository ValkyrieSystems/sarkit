# Contributing to SARkit

SARkit is a Python package that is intended to simplify working with the NGA SAR
standards.

## Getting started with development

### Setup

#### Clone the repository

```bash
git clone git@github.com:ValkyrieSystems/sarkit.git
cd sarkit
```

#### Create and activate an environment

The default and supported way to create an environment is creating
a python virtual environment.

```bash
python -m venv ./venv
source ./venv/bin/activate
```

Other environment managers, such as conda, may be used to create and activate
an environment.  For simplicity, examples will use the built-in Python virtual
environments.

#### Package installation

Once an environment is created and activated, you can install the packages
into your environment with pip from the package root.

```bash
python -m pip install .
```

The following install command would install core dependencies and allow for linting and testing.
mypy works best with an editable install.  Requires pip >= 25.1

```bash
python -m pip install --editable . --group dev
```

It is recommended that, whenever possible, contributors install all optional
dependencies so that all unit tests can be run prior to submission of a
Pull Request.

### Running tests

In the development environment, unit tests can be run with pytest.

```bash
pytest tests
```

Unit tests are organized by functional group so that the unit tests for the core functionality
can be run as below.

```bash
pytest tests/core
```

## Submitting changes

Even more excellent than a good bug report is a fix for a bug, or the
implementation of a much-needed new feature. We'd love to have
your contributions.

### Pre-PR checking

Prior to submitting your contribution through a Pull Request, the lint and
unit tests should be run and pass cleanly.

The linting will require the `dev-lint` dependencies to be installed and can
be done with the following:

```bash
ruff check
mypy sarkit
```

If the linting does not pass cleanly, the following commands can be used to
automatically fix some formatting errors.

```bash
ruff check --fix
ruff format
```

### Pull Request process

We use the usual GitHub pull-request flow, which may be familiar to
you if you've contributed to other projects on GitHub.  For the mechanics,
see [GitHub's documentation](https://help.github.com/articles/using-pull-requests/).

Anyone interested in SARkit may review your code.  One of the core
developers will merge your pull request when they think it's ready.

If your change will be a significant amount of work
to write, we highly recommend starting by opening an issue laying out
what you want to do.  That lets a conversation happen early in case
other contributors disagree with what you'd like to do or have ideas
that will help you do it.

The best pull requests are focused, clearly describe what they're for
and why they're correct, and contain tests for whatever changes they
make to the code's behavior.  As a bonus these are easiest for someone
to review, which helps your pull request get merged quickly!  Standard
advice about good pull requests for open-source projects applies.

Also, do not squash your commits after you have submitted a pull request, as this
erases context during review. We will squash commits when the pull request is merged.
