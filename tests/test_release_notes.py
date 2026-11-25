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


URL = "https://xoa.readthedocs.io/v2.0.0/"


@pytest.fixture
def changes(tmp_path):
    path = tmp_path / "CHANGES.rst"
    path.write_text(CHANGES)
    return path


@pytest.fixture
def root(tmp_path):
    """A documentation tree with labels, an include and tutorials"""
    doc = tmp_path / "doc"
    doc.mkdir()
    (doc / "indepth.plot.rst").write_text(
        ".. _indepth.plot:\n\nPlotting\n########\n\nText.\n\n"
        ".. _indepth.plot.maps:\n\nMaps\n====\n\nMore text.\n"
    )
    (doc / "contributing.rst").write_text(".. include:: ../CONTRIBUTING.rst\n")
    (tmp_path / "CONTRIBUTING.rst").write_text(".. _contributing:\n\nContributing\n============\n")
    (doc / "configure.rst").write_text("Configuration\n#############\n")
    gallery = tmp_path / "tutorials" / "grids"
    gallery.mkdir(parents=True)
    (tmp_path / "tutorials" / "README.rst").write_text(".. _tutorials:\n\nTutorials\n=========\n")
    (gallery / "plot_geosel.py").write_text('"""\nSelect data\n===========\n\nText.\n"""\n')
    release_notes.get_labels.cache_clear()
    return tmp_path


def test_section_only_and_markdown(changes, tmp_path):
    notes = release_notes.get_release_notes("2.0.0", changes, root=tmp_path)
    assert notes == (
        "### New features\n"
        "- Add `xoa.plot.plot_vectors` and `plot_field`, see indepth.plot.\n"
        "- Use `nsigma` like in the guide and `xoa.cf.CFSpecs.decode`.\n"
        "\n"
        "### Bug fixes\n"
        "- Fix a bug [#12], see #3.\n"
    )


def test_doc_links(changes, root):
    notes = release_notes.get_release_notes("2.0.0", changes, root=root)
    assert f"see [Plotting]({URL}indepth.plot.html)." in notes
    assert f"like in [the guide]({URL}indepth.plot.html) and" in notes


@pytest.mark.parametrize(
    "role, title, url",
    [
        (":ref:`indepth.plot.maps`", "Maps", "indepth.plot.html#indepth-plot-maps"),
        (":ref:`contributing`", "Contributing", "contributing.html"),
        (":doc:`configure`", "Configuration", "configure.html"),
        (":doc:`/configure`", "Configuration", "configure.html"),
        (":ref:`tutorials`", "Tutorials", "tutorials/index.html"),
        (
            ":ref:`sphx_glr_tutorials_grids_plot_geosel.py`",
            "Select data",
            "tutorials/grids/plot_geosel.html",
        ),
        (
            ":ref:`Geosel <sphx_glr_tutorials_grids_plot_geosel.py>`",
            "Geosel",
            "tutorials/grids/plot_geosel.html",
        ),
    ],
)
def test_doc_link_targets(root, role, title, url):
    notes = release_notes.to_markdown([role], release_notes.get_labels(root), URL)
    assert notes == f"[{title}]({URL}{url})\n"


def test_unknown_label(root):
    notes = release_notes.to_markdown([":ref:`missing.label`"], release_notes.get_labels(root), URL)
    assert notes == "missing.label\n"


def test_unreleased_links_to_develop(tmp_path, root):
    path = tmp_path / "CHANGES.rst"
    path.write_text("Unreleased\n==========\n\n- See :ref:`indepth.plot`.\n")
    notes = release_notes.get_release_notes("Unreleased", path, root=root)
    assert notes == "- See [Plotting](https://xoa.readthedocs.io/develop/indepth.plot.html).\n"


def test_repository_refs_resolve():
    """All the refs of the real changes file have a title"""
    release_notes.get_labels.cache_clear()
    labels = release_notes.get_labels()
    text = release_notes.CHANGES.read_text()
    for role, content in release_notes.ROLE.findall(text):
        if role in ("ref", "doc"):
            target = content.rsplit("<", 1)[-1].rstrip(">").lstrip("/")
            assert (role, target) in labels, target


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
