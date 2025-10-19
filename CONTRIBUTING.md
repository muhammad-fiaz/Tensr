# Contributing to Tensr

Thank you for your interest in contributing to Tensr! This document provides guidelines for contributing.

## Code of Conduct

Be respectful and inclusive. We welcome contributions from everyone.

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported
2. Create a new issue with:
   - Clear title and description
   - Steps to reproduce
   - Expected vs actual behavior
   - System information

### Suggesting Features

1. Check if the feature has been suggested
2. Create an issue describing:
   - Use case
   - Proposed API
   - Implementation ideas

### Pull Requests

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Update documentation
6. Submit a pull request

## Development Setup

```bash
git clone https://github.com/muhammad-fiaz/tensr.git
cd tensr
xmake build
xmake run tests
```

## Coding Standards

- Follow C11/C++17 standards
- Use meaningful variable names
- Add comments for complex logic
- Write unit tests for new features
- Keep functions small and focused

## Testing

All tests must pass:

```bash
xmake run tests
```

## Documentation

Update documentation for:
- New features
- API changes
- Examples

## Commit Messages

Use clear, descriptive commit messages:

```
Add matrix multiplication optimization

- Implement SIMD vectorization
- Add benchmarks
- Update documentation
```

## License

By contributing, you agree that your contributions will be licensed under Apache License 2.0.

## Questions?

Contact: contact@muhammadfiaz.com
