# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Ability to read and write SIDD DED segments
- SICD and SIDD consistency checks for ValidData bounds
- Support for SIDD v1.0

### Changed
- Improved read performance
- Pinned `jbpy` dependency

### Fixed
- SICD consistency failure when optional IPP Sets were omitted
- `ElementWrapper` not creating non-repeatable subelement for empty dict


## [1.1.0] - 2025-09-29

### Added
- `sarkit.cphd.Reader` support for reading a subset of vectors
- New `check_*`methods for SICD consistency
- More complete unit tests for SICD consistency
- `ElementWrapper.get` now supports the `default` parameter

### Changed
- `ElementWrapper.__contains__` now raises `KeyError` for impossible keys

### Fixed
- Minor bugfixes for existing SICD consistency `check_*` methods


## [1.0.1] - 2025-09-09

### Fixed
- Use most recent SICD v1.4.0 XML schema (dated 2024-05-01)


## [1.0.0] - 2025-09-04

### Added
- CPHD reference geometry calculations
- CRSD reference geometry and related computations

### Changed
- CLIs renamed from `<>-consistency` to `<>check`
- Rearranged CRSD reference geometry API
- Rearranged SICD sensitivity/error propagation API

### Removed
- Several incomplete / NotImplemented APIs


## [0.12.0] - 2025-08-18

### Added
- Support for reading a sub-image from a SICD NITF

### Fixed
- Handling of non-UTC datetime fields in SICD and SIDD


## [0.11.0] - 2025-08-11

### Added
- Support for writing masked arrays in CRSD
- CLI utilities now use `smart_open` if it is installed
- Added `crsdinfo` CLI utility
- Added `cphdinfo` CLI utility
- Added `sicdinfo` CLI utility
- Added `siddinfo` CLI utility

### Changed
- Simplified some CPHD consistency checks


## [0.10.0] - 2025-07-16

### Added
- ElementWrapper classes for interacting with SAR XML
- Improved handling of Compressed CRSD
- Support for using `smart_open` to open remote files

### Fixed
- SICD consistency checker's SegmentList checks


## [0.9.0] - 2025-07-01

### Changed
- Replaced built in NITF parsing with `jbpy`


## [0.8.0] - 2025-06-18

### Added
- CRSD v1.0 Reading and Writing
- SICD RIC frame handling

### Removed
- Support for CRSD v1.0 DRAFT


## [0.7.0] - 2025-05-26

### Added
- SIDD geometry calculations
- SIDD consistency checker


## [0.6.1] - 2025-05-07

### Added
- Support for python 3.13
- SICD segmentation check to consistency checker

### Fixed
- Writing CPHDs without support arrays
- SICD projection to a DEM surface not iterating enough


## [0.6.0] - 2025-04-16

### Added
- Support for `numpy` >= 1.25
- SICD projeciton to a DEM surface
- SICD sensitivity matricies calculations

### Changed
- Improved consistency checker APIs


## [0.5.0] - 2025-04-02

### Changed
- Improved SICD projection API


## [0.4.0] - 2025-03-25

### Added
- Support for SIDD MONO8LU and RGB8LU pixel types
- Partial support for compressed CRSD

### Changed
- Even flatter API structure


## [0.3.0] - 2025-03-13

### Changed
- Further API refinements


## [0.2.0] - 2025-02-20

### Added
- Support for APOs in SICD projections
- Better handling of SIDD MONO16I and RGB24I pixel types

### Changed
- Reworked API to have flatter structure


## [0.1.0] - 2025-01-22

### Added
- Limited CRSD DRAFT Reading and Writing
- Limited CPHD Reading and Writing
- Limited SICD Reading and Writing
- Limited SIDD NITF Reading and Writing

[unreleased]: https://github.com/ValkyrieSystems/sarkit/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/ValkyrieSystems/sarkit/compare/v1.0.1...v1.1.0
[1.0.1]: https://github.com/ValkyrieSystems/sarkit/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/ValkyrieSystems/sarkit/compare/v0.12.0...v1.0.0
[0.12.0]: https://github.com/ValkyrieSystems/sarkit/compare/v0.11.0...v0.12.0
[0.11.0]: https://github.com/ValkyrieSystems/sarkit/compare/v0.10.0...v0.11.0
[0.10.0]: https://github.com/ValkyrieSystems/sarkit/compare/v0.9.0...v0.10.0
[0.9.0]: https://github.com/ValkyrieSystems/sarkit/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/ValkyrieSystems/sarkit/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/ValkyrieSystems/sarkit/compare/v0.6.1...v0.7.0
[0.6.1]: https://github.com/ValkyrieSystems/sarkit/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/ValkyrieSystems/sarkit/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/ValkyrieSystems/sarkit/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/ValkyrieSystems/sarkit/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/ValkyrieSystems/sarkit/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/ValkyrieSystems/sarkit/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/ValkyrieSystems/sarkit/releases/tag/v0.1.0
