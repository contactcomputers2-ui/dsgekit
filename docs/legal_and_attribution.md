# Legal and Attribution Notes

This document provides a public legal notice and third-party attribution references.

It is not legal advice.

## Non-affiliation Notice

- `dsgekit` is an independent project.
- References to third-party tools are descriptive and limited to compatibility or migration context.
- The project is not affiliated with, endorsed by, or sponsored by third-party tool maintainers.

## Implementation Provenance Policy

- Repository code is intended to be independently implemented.
- Third-party names may be referenced for interoperability context only.
- Third-party source code must not be copied into this repository unless licensing is compatible and attribution obligations are explicitly met.
- Substantial third-party documentation text must not be copied into public docs.
- Contributions are governed by `CONTRIBUTING.md` originality/provenance requirements.

## Third-Party Reference Snapshot

Checked on: 2026-02-07

- Dynare project software license: GNU GPL v3 or later  
  https://www.dynare.org/license/
- Manual license: GNU Free Documentation License 1.3  
  https://www.dynare.org/manual/index.html#preface
- Website content license: CC BY-NC-ND 4.0 (site footer)  
  https://www.dynare.org/

## Scope

- This page does not include internal release procedures.
- Internal release gate checklist: `docs/release_legal_gate.md`.
- Public references to third-party tools in this repository are intended only for interoperability and migration context.

## Third-Party Dependency Notices

- Inventory file: `THIRD_PARTY_NOTICES.md` (repository root).
- Generation script: `tools/generate_third_party_notices.py`.
- Rebuild command:

```bash
python tools/generate_third_party_notices.py --extras sym,plot,speed,jax
```

Notes:

- The inventory is generated from `pyproject.toml` plus installed package metadata.
- Rows marked `missing` are declared in the selected dependency closure but not installed in the current environment.

## Naming Policy

- Internal wording policy for third-party terms: `docs/brand_naming_policy.md`.
