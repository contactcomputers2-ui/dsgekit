# Contributing to dsgekit

Thanks for your interest in contributing.

## Ground Rules

- Be respectful and constructive.
- Prefer small, focused pull requests.
- Open an issue first for large changes or architecture proposals.
- Keep discussions in English or Spanish.

## Ways to Contribute

- Report bugs.
- Propose features.
- Improve docs and examples.
- Submit code fixes and tests.

## Reporting Issues

Please include:

- What you expected to happen.
- What happened instead.
- Reproduction steps.
- Minimal model or snippet (if possible).
- Python version and OS.

For security-related issues, do not open a public issue. See `SECURITY.md`.

## Pull Requests

PRs are welcome.

Before opening a PR:

1. Create a branch from `main`.
2. Add or update tests for behavior changes.
3. Run checks locally:
   - `.venv/bin/ruff check src tests`
   - `.venv/bin/pytest -q`
4. Update docs/changelog when relevant.
5. Fill the PR template clearly.

## Originality and Provenance

Contributions must be original work, or clearly identified third-party material with compatible licensing and attribution.

- Do not copy source code from third-party repositories unless the license is compatible and attribution requirements are satisfied.
- Do not paste substantial text from third-party manuals, docs, or tutorials into this repository.
- If behavior is based on a public specification/manual, implement it independently and document assumptions in the PR description.
- If you adapt external snippets, include exact provenance and license details in the PR.

## Development Setup

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,plot]"
```

## Scope and Compatibility

- Project status is alpha (`0.1.0.dev*` pre-release cycle).
- Current minimum Python version is defined in `pyproject.toml`.
- Backward-compatibility for APIs is not guaranteed yet.

## License

By contributing, you agree that your contributions are licensed under the MIT License in this repository.
