"""Prompt template loader with caching and strategy-specific post-processing."""

import json
import os
import importlib.util
import sys
from pathlib import Path
from functools import lru_cache
from typing import Union, Dict, Callable


# Get the directories
TEMPLATES_DIR = Path(__file__).parent / "templates"
STRATEGIES_DIR = Path(__file__).parent / "strategies"


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


def load_prompt(strategy_name: str, problem: str, api_mode: str = "completion") -> Union[str, Dict[str, str]]:
    """
    Load a prompt template and substitute the problem.

    Tries new strategies folder first, falls back to templates folder.

    Args:
        strategy_name: Name of the prompt strategy (e.g., 'minimal', 'fewshot_v1')
        problem: The HumanEval problem (function signature + docstring)
        api_mode: API mode - "completion" or "chat"

    Returns:
        - For completion mode: string with problem substituted
        - For chat mode: dict with 'system' and 'user' keys

    Raises:
        FileNotFoundError: If the strategy template doesn't exist
    """
    if api_mode == "chat":
        # Try new strategies folder first
        strategy_path = STRATEGIES_DIR / "chat" / strategy_name / "prompt.json"

        if not strategy_path.exists():
            # Fall back to old templates folder
            strategy_path = TEMPLATES_DIR / "chat" / f"{strategy_name}.json"

        if not strategy_path.exists():
            raise FileNotFoundError(
                f"Chat prompt template '{strategy_name}' not found. "
                f"Available templates: {list_available_strategies(api_mode)}"
            )

        template_content = _load_template(str(strategy_path))
        template = json.loads(template_content)

        # Substitute {problem} in the user message
        return {
            "system": template["system"],
            "user": template["user"].format(problem=problem)
        }
    else:
        # Try new strategies folder first
        strategy_path = STRATEGIES_DIR / "completion" / strategy_name / "prompt.txt"

        if not strategy_path.exists():
            # Fall back to old templates folder
            strategy_path = TEMPLATES_DIR / "completion" / f"{strategy_name}.txt"

        if not strategy_path.exists():
            raise FileNotFoundError(
                f"Completion prompt template '{strategy_name}' not found. "
                f"Available templates: {list_available_strategies(api_mode)}"
            )

        template = _load_template(str(strategy_path))
        return template.format(problem=problem)


def load_post_processor(strategy_name: str, api_mode: str = "completion") -> Callable:
    """
    Load the post-processing function for a specific strategy.

    Args:
        strategy_name: Name of the prompt strategy
        api_mode: API mode - "completion" or "chat"

    Returns:
        Post-processing function (callable)

    Raises:
        FileNotFoundError: If no post-processor exists for this strategy
    """
    # Look for post_process.py in the strategy folder
    if api_mode == "chat":
        post_process_path = STRATEGIES_DIR / "chat" / strategy_name / "post_process.py"
    else:
        post_process_path = STRATEGIES_DIR / "completion" / strategy_name / "post_process.py"

    if not post_process_path.exists():
        raise FileNotFoundError(
            f"No post-processor found for strategy '{strategy_name}' at {post_process_path}"
        )

    # Dynamically load the module
    spec = importlib.util.spec_from_file_location(
        f"post_process_{strategy_name}",
        post_process_path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    # Return the post_process function
    if not hasattr(module, 'post_process'):
        raise AttributeError(
            f"Post-processor for '{strategy_name}' must have a 'post_process' function"
        )

    return module.post_process


def list_available_strategies(api_mode: str = "completion") -> list[str]:
    """
    List all available prompt strategies for the given API mode.

    Includes both new strategies folder and old templates folder.

    Args:
        api_mode: API mode - "completion" or "chat"

    Returns:
        List of strategy names (without file extensions)
    """
    strategies = set()

    # Check new strategies folder
    if api_mode == "chat":
        strategies_path = STRATEGIES_DIR / "chat"
        if strategies_path.exists():
            for folder in strategies_path.iterdir():
                if folder.is_dir() and (folder / "prompt.json").exists():
                    strategies.add(folder.name)

        # Fall back to old templates
        templates_path = TEMPLATES_DIR / "chat"
        if templates_path.exists():
            for file in templates_path.glob("*.json"):
                strategies.add(file.stem)
    else:
        strategies_path = STRATEGIES_DIR / "completion"
        if strategies_path.exists():
            for folder in strategies_path.iterdir():
                if folder.is_dir() and (folder / "prompt.txt").exists():
                    strategies.add(folder.name)

        # Fall back to old templates
        templates_path = TEMPLATES_DIR / "completion"
        if templates_path.exists():
            for file in templates_path.glob("*.txt"):
                strategies.add(file.stem)

    return sorted(list(strategies))


def clear_cache():
    """Clear the template cache (useful for development/testing)."""
    _load_template.cache_clear()
