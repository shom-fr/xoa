#!/usr/bin/env python3
"""Print the release notes of a version of CHANGES.rst as Markdown

Usage: python bin/release_notes.py 2026.10.0 | gh release create v2026.10.0 --notes-file -
"""

import re
import sys
from pathlib import Path

CHANGES = Path(__file__).resolve().parent.parent / "CHANGES.rst"

ROLE = re.compile(r":(\w+):`([^`]+)`")
UNDERLINE = re.compile(r"^([=\-#~^])\1+$")


def get_section(text, version):
    """Lines of the section titled `version`, without its title"""
    lines = text.splitlines()
    start = None
    for i in range(len(lines) - 1):
        if UNDERLINE.match(lines[i + 1]) and lines[i].strip():
            if start is not None and lines[i + 1][0] == "=":
                return lines[start:i]
            if lines[i] == version and lines[i + 1][0] == "=":
                start = i + 2
    if start is None:
        raise ValueError(f"Version not found in CHANGES.rst: {version}")
    return lines[start:]


def convert_role(match):
    """Markdown of a Sphinx role: a link to an issue or pull request, or code or text"""
    role, content = match.groups()
    if role in ("pull", "issue"):
        return f"#{content}"
    label = re.match(r"(.+?)\s*<[^>]+>$", content)
    if label:
        return label.group(1)
    if role in ("ref", "doc"):
        return content
    return f"`{content[1:].rsplit('.', 1)[-1] if content.startswith('~') else content}`"


def to_markdown(lines):
    """Convert reStructuredText lines of a section to Markdown"""
    out = []
    for i, line in enumerate(lines):
        if UNDERLINE.match(line):
            continue
        if i + 1 < len(lines) and UNDERLINE.match(lines[i + 1]) and line.strip():
            out.append(f"### {line}")
            continue
        line = ROLE.sub(convert_role, line)
        out.append(re.sub(r"``([^`]+)``", r"`\1`", line))
    return "\n".join(out).strip() + "\n"


def get_release_notes(version, path=CHANGES):
    """Markdown release notes of a version of the changes file"""
    return to_markdown(get_section(Path(path).read_text(), version.removeprefix("v")))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        sys.exit(__doc__)
    try:
        sys.stdout.write(get_release_notes(argv[0]))
    except ValueError as exc:
        sys.exit(str(exc))


if __name__ == "__main__":
    main()
