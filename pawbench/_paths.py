"""Shared contract primitives used by more than one subpackage.

Internal module: not part of the public interface.
"""

from __future__ import annotations


def safe_relative_path_detail(value: object) -> str | None:
    """Return a problem detail if ``value`` is not a safe package-relative path.

    Safe means: a non-empty POSIX relative path with no ``.``/``..`` segments,
    no drive letters, no URI schemes, no backslashes, and no control
    characters.
    """
    if not isinstance(value, str) or value == "":
        return "path must be a non-empty string"
    if value.startswith("/"):
        return "path must be relative (found leading '/')"
    if "\\" in value:
        return "path must use POSIX separators (found '\\')"
    if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in value):
        return "path must not contain control characters"
    first_segment = value.split("/")[0]
    if ":" in first_segment:
        return "path must not contain ':' in the first segment (URI scheme or drive letter)"
    for segment in value.split("/"):
        if segment == "":
            return "path must not contain empty segments ('//' or trailing '/')"
        if segment in (".", ".."):
            return f"path must not contain {segment!r} segments"
    return None
