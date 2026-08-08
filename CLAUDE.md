# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project is a Python CLI request classifier using pytest for testing. It classifies incoming requests into predefined categories based on their content.

## Common Commands

### Development

- Run the classifier: `python main.py`
- Run tests: `pytest`
- Run a single test: `pytest tests/test_classifier.py::test_classify_request`
- Lint the code: `flake8`
- Format the code: `black .`

### Project Structure

The project follows a standard Python package structure:

- `main.py`: Entry point for the CLI application
- `classifier/`: Core classification logic
  - `classifier.py`: Main classification module
  - `models.py`: Data models and schemas
- `tests/`: Test suite
  - `test_classifier.py`: Unit tests for the classifier
  - `test_models.py`: Unit tests for data models
- `data/`: Sample data for testing and development

## Style Guide

- Follow PEP 8 guidelines for Python code
- Use type hints for all functions and methods
- Keep line length under 88 characters
- Use descriptive variable and function names
- Write docstrings for all modules, classes, and functions

## Testing

- All new features must include corresponding unit tests
- Use pytest fixtures for test data
- Maintain test coverage above 80%
- Run tests before submitting any changes

## Development Practices

- Commit frequently with clear, descriptive messages
- Write meaningful commit messages that explain the "why" behind changes
- Review your own code before creating a pull request
- Keep pull requests small and focused on a single feature or bug fix

## Continuous Integration

The project uses GitHub Actions for CI. The workflow includes:

- Linting with flake8
- Formatting with black
- Unit tests with pytest
- Code coverage reporting

## Documentation

- Update the README.md with any new features or changes
- Document public APIs and their usage
- Include examples in the documentation
- Keep documentation up-to-date with code changes

## Troubleshooting

- If tests are failing, check the error messages for clues
- Use print statements or logging for debugging
- Consult the project's issue tracker for known problems
- Ask for help in the project's communication channels if needed