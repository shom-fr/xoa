#!/usr/bin/env python3
"""Print the release notes of a version of CHANGES.rst as Markdown

Usage: python bin/release_notes.py 2026.10.0 | gh release create v2026.10.0 --notes-file -

The ``:ref:`` and ``:doc:`` roles become links to the documentation of the version
on Read the Docs, titled like the sections and tutorials of ``doc`` and ``tutorials``.
"""

import ast
import functools
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHANGES = ROOT / "CHANGES.rst"
DOC_URL = "https://xoa.readthedocs.io/{version}/"

ROLE = re.compile(r":(\w+):`([^`]+)`")
UNDERLINE = re.compile(r"^([=\-#~^*+])\1+$")
LABEL = re.compile(r"^\.\. _([^:]+):\s*$")
INCLUDE = re.compile(r"^\.\. include:: (\S+\.rst)\s*$", re.M)


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


def is_title(lines, i):
    """Whether line `i` is a section title, underlined, with or without an overline"""
    return (
        i + 1 < len(lines)
        and lines[i].strip() != ""
        and not UNDERLINE.match(lines[i])
        and UNDERLINE.match(lines[i + 1]) is not None
    )


def make_id(label):
    """HTML anchor of a label, like docutils"""
    return re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")


def read_rst(path):
    """Text of a reStructuredText file, with its included files"""
    return INCLUDE.sub(lambda m: read_rst(path.parent / m.group(1)), path.read_text())


def scan_rst(text, page, labels, docname=None):
    """Store the titled links of the labels of a reStructuredText page in `labels`"""
    lines = text.splitlines()
    titles = [i for i in range(len(lines)) if is_title(lines, i)]
    if docname and titles:
        labels["doc", docname] = (lines[titles[0]].strip(), page)
    for j, line in enumerate(lines):
        match = LABEL.match(line)
        following = [i for i in titles if i > j]
        if not match or not following:
            continue
        anchor = f"#{make_id(match.group(1))}" if any(i < j for i in titles) else ""
        labels["ref", match.group(1)] = (lines[following[0]].strip(), page + anchor)


@functools.cache
def get_labels(root=ROOT):
    """Titles and relative URLs of the labels and documents, keyed by (role, target)"""
    root = Path(root)
    labels = {}
    for path in sorted((root / "doc").glob("*.rst")):
        scan_rst(read_rst(path), f"{path.stem}.html", labels, path.stem)
    tutorials = root / "tutorials"
    for path in sorted(tutorials.glob("**/README.rst")):
        page = path.parent.relative_to(root).as_posix() + "/index.html"
        scan_rst(path.read_text(), page, labels)
    for path in sorted(tutorials.glob("*/*.py")):
        docstring = ast.get_docstring(ast.parse(path.read_text())) or ""
        lines = docstring.splitlines()
        titles = [lines[i].strip() for i in range(len(lines)) if is_title(lines, i)]
        if titles:
            label = f"sphx_glr_tutorials_{path.parent.name}_{path.name}"
            page = f"tutorials/{path.parent.name}/{path.stem}.html"
            labels["ref", label] = (titles[0], page)
    return labels


def convert_role(match, labels=None, base_url=None):
    """Markdown of a Sphinx role: a link to an issue, a pull request or the doc, or code"""
    role, content = match.groups()
    if role in ("pull", "issue"):
        return f"#{content}"
    explicit = re.match(r"(.+?)\s*<([^>]+)>$", content)
    text, target = explicit.groups() if explicit else (None, content)
    if role in ("ref", "doc"):
        key = role, target.lstrip("/") if role == "doc" else target
        if labels and base_url and key in labels:
            title, page = labels[key]
            return f"[{text or title}]({base_url}{page})"
        return text or target
    if text:
        return text
    return f"`{content[1:].rsplit('.', 1)[-1] if content.startswith('~') else content}`"


def to_markdown(lines, labels=None, base_url=None):
    """Convert reStructuredText lines of a section to Markdown"""
    out = []
    convert = functools.partial(convert_role, labels=labels, base_url=base_url)
    for i, line in enumerate(lines):
        if UNDERLINE.match(line):
            continue
        if i + 1 < len(lines) and UNDERLINE.match(lines[i + 1]) and line.strip():
            out.append(f"### {line}")
            continue
        line = ROLE.sub(convert, line)
        out.append(re.sub(r"``([^`]+)``", r"`\1`", line))
    return "\n".join(out).strip() + "\n"


def get_release_notes(version, path=CHANGES, root=ROOT):
    """Markdown release notes of a version of the changes file

    The doc links point to the ``v<version>`` documentation, or to ``develop``
    for the unreleased changes.
    """
    version = version.removeprefix("v")
    base_url = DOC_URL.format(version="develop" if version == "Unreleased" else f"v{version}")
    lines = get_section(Path(path).read_text(), version)
    return to_markdown(lines, get_labels(Path(root)), base_url)


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
