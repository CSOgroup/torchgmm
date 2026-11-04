# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [Unreleased]

### Added

- Basic tool, preprocessing and plotting functions
- Support for Python 3.14 ([#24](https://github.com/CSOgroup/torchgmm/issues/24))

### Fixed

- Import error on Python 3.14 caused by setting `__doc__` on `typing.Union` type aliases
