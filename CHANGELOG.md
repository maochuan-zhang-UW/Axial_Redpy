# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

**Pending code review and approval**

This version will be the official public release of REDPy and the start of
semantic versioning. This version is a massive overhaul of the original code
hosted on [GitHub](https://github.com/ahotovec/REDPy), the last commit of which
I reference here and in the documentation as version 0. The full commit history
of the code from its inception in 2014 has been preserved in this repository.
Changes from version 0 are too numerous to list in detail, but the major
features are noted below.

### Changed

- Move to object-oriented programming
- Console scripts (`redpy-*`) replace `*.py` scripts in root directory
- Performance improvements
- Introduction of several new configuration settings
- Unit and coverage tests added
- Minor processing changes that break true backward compatibility
- Retains compatibility with old `.h5` and `.cfg` files
