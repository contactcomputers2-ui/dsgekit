"""Unified model loading interface.

Provides a single entry point for loading DSGE models from any supported format.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dsgekit.model.calibration import Calibration
    from dsgekit.model.ir import ModelIR
    from dsgekit.model.steady_state import SteadyState


def load_model(
    source: str | Path | dict,
    format: str | None = None,
) -> tuple[ModelIR, Calibration, SteadyState]:
    """Load a DSGE model from file or dict.

    Automatically detects format based on file extension:
    - .mod: `.mod` style format
    - .yaml, .yml: YAML format
    - dict: Python dictionary (YAML-like structure)

    Args:
        source: File path or dictionary
        format: Override format detection ('mod', 'yaml', 'dict')

    Returns:
        Tuple of (ModelIR, Calibration, SteadyState)

    Raises:
        ValueError: If format cannot be determined
        FileNotFoundError: If file does not exist
        ParseError: If parsing fails

    Examples:
        # From .mod file
        model, cal, ss = load_model("model.mod")

        # From YAML file
        model, cal, ss = load_model("model.yaml")

        # From dict
        model, cal, ss = load_model({
            "name": "AR1",
            "variables": ["y"],
            "shocks": ["e"],
            "parameters": {"rho": 0.9},
            "equations": ["y = rho * y(-1) + e"],
            "steady_state": {"y": 0},
            "shocks_config": {"e": {"stderr": 0.01}},
        })
    """

    # Handle dict input
    if isinstance(source, dict):
        if format and format != "dict":
            raise ValueError(f"Dict input but format='{format}' specified")
        from dsgekit.io.formats.yaml_format import _parse_yaml_content
        return _parse_yaml_content(source)

    # Handle file path
    path = Path(source)

    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    # Determine format
    if format is None:
        suffix = path.suffix.lower()
        if suffix == ".mod":
            format = "mod"
        elif suffix in (".yaml", ".yml"):
            format = "yaml"
        else:
            raise ValueError(
                f"Cannot determine format from extension '{suffix}'. "
                "Use format='mod' or format='yaml' explicitly."
            )

    # Load based on format
    if format == "mod":
        from dsgekit.io.formats.mod import load_mod_file
        return load_mod_file(str(path))

    elif format == "yaml":
        from dsgekit.io.formats.yaml_format import load_yaml
        return load_yaml(path)

    else:
        raise ValueError(f"Unknown format: '{format}'. Supported: 'mod', 'yaml'")


def load(source: str | Path | dict, **kwargs) -> tuple[ModelIR, Calibration, SteadyState]:
    """Alias for load_model."""
    return load_model(source, **kwargs)
