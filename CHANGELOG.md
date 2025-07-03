# Changelog

All notable changes to the vCenter DRS Compliance Dashboard project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- GitHub templates for issues and pull requests
- Code of Conduct (Contributor Covenant)
- Contributing guidelines
- Security policy
- Changelog tracking

### Changed
- Improved project documentation structure

## [1.0.0] - 2024-01-XX

### Added
- Initial release of vCenter DRS Compliance Dashboard
- Streamlit web application for DRS compliance monitoring
- Database layer with support for PostgreSQL, MySQL, and SQLite
- DRS compliance rules engine
- vCenter data collection and analysis
- Real-time compliance metrics and visualizations
- Automated data refresh capabilities
- Comprehensive test suite
- GitHub Actions CI/CD pipeline
- Type checking with mypy
- Code quality tools (black, isort, flake8)

### Features
- **Dashboard**: Real-time DRS compliance monitoring
- **Data Collection**: Automated vCenter data gathering
- **Compliance Rules**: Configurable DRS compliance checks
- **Database Support**: Multiple database backends
- **Metrics**: Prometheus-compatible metrics export
- **Scheduling**: Automated data refresh with cron/systemd
- **Configuration**: Environment-based configuration management

### Technical Details
- **Language**: Python 3.8+
- **Framework**: Streamlit
- **Database**: PostgreSQL, MySQL, SQLite
- **Testing**: pytest with coverage
- **CI/CD**: GitHub Actions
- **Code Quality**: mypy, black, isort, flake8

## [0.1.0] - 2024-01-XX

### Added
- Initial development version
- Basic vCenter connectivity
- Simple DRS compliance checks
- Streamlit interface prototype

---

## Version History

- **1.0.0**: First stable release with full feature set
- **0.1.0**: Initial development version

## Release Process

### Version Numbering

This project follows [Semantic Versioning](https://semver.org/):

- **MAJOR** version for incompatible API changes
- **MINOR** version for backwards-compatible functionality additions
- **PATCH** version for backwards-compatible bug fixes

### Release Checklist

Before each release:

- [ ] All tests pass
- [ ] Documentation is updated
- [ ] Changelog is updated
- [ ] Version numbers are updated in code
- [ ] Security review completed
- [ ] Performance testing completed
- [ ] Release notes prepared

### Release Notes

Release notes are automatically generated from this changelog and published with each GitHub release.

---

## Contributing to the Changelog

When contributing to this project, please update the changelog by adding entries under the `[Unreleased]` section following the format above.

### Changelog Entry Types

- **Added**: New features
- **Changed**: Changes in existing functionality
- **Deprecated**: Soon-to-be removed features
- **Removed**: Removed features
- **Fixed**: Bug fixes
- **Security**: Vulnerability fixes

### Example Entry

```markdown
### Added
- New feature for enhanced DRS monitoring
- Additional compliance rule types

### Changed
- Updated database connection handling
- Improved error messages

### Fixed
- Resolved issue with vCenter authentication
- Fixed data refresh scheduling bug
``` 