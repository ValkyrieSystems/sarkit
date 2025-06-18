<div align="center">

<img src="https://raw.githubusercontent.com/ValkyrieSystems/sarkit/main/docs/source/_static/sarkit_logo.png" width=200>

[![PyPI - Version](https://img.shields.io/pypi/v/sarkit)](https://pypi.org/project/sarkit/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/sarkit)
[![PyPI - License](https://img.shields.io/pypi/l/sarkit)](./LICENSE)
[![SPEC 0 — Minimum Supported Dependencies](https://img.shields.io/badge/SPEC-0-green?labelColor=%23004811&color=%235CA038)](https://scientific-python.org/specs/spec-0000/)
<br>
[![Tests](https://github.com/ValkyrieSystems/sarkit/actions/workflows/tests.yml/badge.svg)](https://github.com/ValkyrieSystems/sarkit/actions/workflows/tests.yml)
[![Documentation Status](https://readthedocs.org/projects/sarkit/badge/?version=latest)](https://sarkit.readthedocs.io/en/latest/?badge=latest)

</div>

**SARkit** is a suite of Synthetic Aperture Radar (SAR)-related tools in Python developed and maintained by
Valkyrie Systems Corporation to encourage the use of National Geospatial-Intelligence Agency (NGA) SAR data standards.

With SARkit, you can:

* read and write SAR standards files (CRSD, CPHD, SICD, SIDD)
* interact with SAR XML metadata using more convenient Python objects
* check SAR data/metadata files for inconsistencies

This project was developed as the modern successor to [SarPy](https://github.com/ngageoint/sarpy).

## License
This repository is licensed under the [MIT license](./LICENSE).

## Contributing and Development
Contributions are welcome; for details see the [contributing guide](./CONTRIBUTING.md).

A few tips for getting started using [PDM](https://pdm-project.org/en/latest/) are below:


```shell
$ pdm install -G:all  # install SARkit with optional & dev dependencies
$ pdm run nox  # run lint and tests
$ pdm run nox -s docs  # build documentation
```
