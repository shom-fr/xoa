Commandline interface
#####################

xoa comes with an executable script with the same name.
Here is its help.

:command:`xoa`
==============

.. argparse::
    :module: xoa.cli
    :func: get_parser
    :prog: xoa
    :nosubcommands:


:command:`xoa info`
===================

.. argparse::
    :module: xoa.cli
    :func: get_parser
    :prog: xoa
    :path: info


Examples
--------

Print all info:

.. command-output:: xoa info


Print only info about paths:

.. command-output:: xoa info paths


:command:`xoa reset_cf_cache`
=============================

.. argparse::
    :module: xoa.cli
    :func: get_parser
    :prog: xoa
    :path: reset_cf_cache


