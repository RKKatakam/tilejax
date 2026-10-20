"""The one exception autokernel raises for code it can't compile."""

from __future__ import annotations

import ast
import linecache

__all__ = ["KernelError"]


class KernelError(Exception):
    """A kernel function autokernel can't compile, or can't run as asked.

    When it is about a line of the user's code, the message starts with `file:line:` and quotes the line.
    """

    def __init__(self, message: str, *, filename: str | None = None, node: ast.AST | None = None) -> None:
        lineno = getattr(node, "lineno", None)
        if filename and lineno:
            message = f"{filename}:{lineno}: {message}"
            line = linecache.getline(filename, lineno).strip()
            if line:
                message += f"\n    {line}"
        super().__init__(message)
