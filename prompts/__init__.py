"""Prompt templates for code generation."""

from .code_completion import (
    create_completion_prompt,
    create_instruction_prompt,
    post_process_completion,
)

__all__ = [
    "create_completion_prompt",
    "create_instruction_prompt",
    "post_process_completion",
]
