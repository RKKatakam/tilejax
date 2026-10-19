"""Exceptions raised by autokernel."""

from __future__ import annotations

__all__ = ["AutoKernelError"]


class AutoKernelError(Exception):
    """Raised when a function can't be turned into a correct Pallas TPU kernel.

    Messages name the function and, where possible, the line in the user's source file.
    """
