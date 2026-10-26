.. _contributing:

Contributing
============

Issues
------

Report bugs and ask for features on the `issue tracker <https://github.com/shom-fr/xoa/issues>`_.
For a bug, give the code that reproduces it and the output of ``xoa info versions``.

Pull requests
-------------

- Branch from ``develop`` and open the pull request against it.
- Install the pre-commit hooks (``pre-commit install``), which run ``ruff`` like the CI.
- Add tests and run ``pytest``.
- Document user visible changes in ``CHANGES.rst`` under ``Unreleased``,
  and list new modules in ``api.rst``.
- Follow the ``CODE_OF_CONDUCT.rst``.

How to release
--------------

The version comes from the git tag (``setuptools-scm``) and follows ``YYYY.M.patch``.

#. Title the ``Unreleased`` changes of ``CHANGES.rst`` with the version,
   and leave an empty ``Unreleased`` above.
#. Merge a ``Version YYYY.M.patch`` pull request from ``develop`` to ``main``.
#. Tag the merge commit and push the tag::

       git tag -a vYYYY.M.patch -m "Version YYYY.M.patch"
       git push origin vYYYY.M.patch

#. Create a GitHub release from the tag, with the notes of ``CHANGES.rst``: it runs the
   workflow that publishes to PyPI::

       python bin/release_notes.py YYYY.M.patch | gh release create vYYYY.M.patch --notes-file -

#. Check Read the Docs and merge the pull request of the
   `conda-forge feedstock <https://github.com/conda-forge/xoa-feedstock>`_.
#. Merge ``main`` back into ``develop``.
