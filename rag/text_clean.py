from __future__ import annotations

import re


def _fix_broken_words_keep_safe(text: str) -> str:
    """
    Fix common PDF extraction artifacts:
    - join split words like "paramet ers" -> "parameters" (conservative)
    - join hyphen line breaks like "real-\n time" -> "real-time"
    """
    if not text:
        return ""

    t = text

    # Fix hyphenated line breaks: "de-\nploy" -> "deploy"
    t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)

    # Fix common split-word artifact: "para met ers" (space inside a word)
    # Conservative: only when both sides are alphabetic and the left part is short.
    t = re.sub(r"\b([A-Za-z]{1,3})\s+([A-Za-z]{2,})\b", r"\1\2", t)

    return t


def normalize_keep_lines(text: str) -> str:
    """
    Clean PDF text but keep line breaks (important for detecting headings/sections).
    """
    if not text:
        return ""

    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = _fix_broken_words_keep_safe(t)

    t = t.replace("\t", " ")

    # Collapse spaces per line, keep line structure
    out_lines = []
    for line in t.split("\n"):
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            out_lines.append(line)

    return "\n".join(out_lines).strip()


def normalize_one_line(text: str) -> str:
    """
    Clean into a single readable line (good for final answers/excerpts).
    """
    if not text:
        return ""
    t = normalize_keep_lines(text)
    t = t.replace("\n", " ")
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"\s+([,.;:])", r"\1", t)
    return t