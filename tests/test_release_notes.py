"""
Test the ``bin/release_notes.py`` script
"""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("release_notes", ROOT / "bin" / "release_notes.py")
release_notes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release_notes)

CHANGES = """What's new
##########

Unreleased
==========

2.0.0
=====

New features
------------
- Add :func:`xoa.plot.plot_vectors` and :func:`~xoa.plot.plot_field`, see :ref:`indepth.plot`.
- Use ``nsigma`` like in :ref:`the guide <indepth.plot>` and :meth:`xoa.cf.CFSpecs.decode`.

Bug fixes
---------
- Fix a bug [:pull:`12`], see :issue:`3`.

1.0.0
=====

Bug fixes
---------
- Old.
"""


@pytest.fixture
def changes(tmp_path):
    path = tmp_path / "CHANGES.rst"
    path.write_text(CHANGES)
    return path


def test_section_only_and_markdown(changes):
    notes = release_notes.get_release_notes("2.0.0", changes)
    assert notes == (
        "### New features\n"
        "- Add `xoa.plot.plot_vectors` and `plot_field`, see indepth.plot.\n"
        "- Use `nsigma` like in the guide and `xoa.cf.CFSpecs.decode`.\n"
        "\n"
        "### Bug fixes\n"
        "- Fix a bug [#12], see #3.\n"
    )


def test_version_prefix_and_last_section(changes):
    assert release_notes.get_release_notes("v1.0.0", changes) == "### Bug fixes\n- Old.\n"


def test_empty_section(changes):
    assert release_notes.get_release_notes("Unreleased", changes) == "\n"


def test_unknown_version(changes):
    with pytest.raises(ValueError, match="9.9"):
        release_notes.get_release_notes("9.9", changes)


def test_main(capsys):
    release_notes.main(["2026.2.0"])
    assert capsys.readouterr().out.startswith("### ")
    with pytest.raises(SystemExit, match="9.9"):
        release_notes.main(["9.9"])
    with pytest.raises(SystemExit):
        release_notes.main([])
