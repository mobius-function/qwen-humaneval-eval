"""Prompt template loader with caching."""

import os
from pathlib import Path
from functools import lru_cache


# Get the templates directory
TEMPLATES_DIR = Path(__file__).parent / "templates"


@lru_cache(maxsize=32)
def _load_template(template_path: str) -> str:
    """
    Load a template file from disk with caching.

    Args:
        template_path: Path to the template file

    Returns:
        Template content as string

    Raises:
        FileNotFoundError: If template doesn't exist
    """
    with open(template_path, 'r', encoding='utf-8') as f:
        return f.read()


def load_prompt(strategy_name: str, problem: str) -> str:
    """
    Load a prompt template and substitute the problem.

    Args:
        strategy_name: Name of the prompt strategy (e.g., 'minimal', 'fewshot_v1')
        problem: The HumanEval problem (function signature + docstring)

    Returns:
        Complete prompt with problem substituted

    Raises:
        FileNotFoundError: If the strategy template doesn't exist
    """
    template_path = TEMPLATES_DIR / f"{strategy_name}.txt"

    if not template_path.exists():
        raise FileNotFoundError(
            f"Prompt template '{strategy_name}' not found at {template_path}. "
            f"Available templates: {list_available_strategies()}"
        )

    template = _load_template(str(template_path))
    return template.format(problem=problem)


def list_available_strategies() -> list[str]:
    """
    List all available prompt strategies.

    Returns:
        List of strategy names (without .txt extension)
    """
    if not TEMPLATES_DIR.exists():
        return []

    templates = []
    for file in TEMPLATES_DIR.glob("*.txt"):
        templates.append(file.stem)  # stem removes the .txt extension

    return sorted(templates)


def clear_cache():
    """Clear the template cache (useful for development/testing)."""
    _load_template.cache_clear()
