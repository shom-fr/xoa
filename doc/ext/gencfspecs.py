"""Generate files to declare and list default CF specifications"""

import os
import logging

import xoa.cf


skip_keys = ["inherit", "processed"]

cat_titles = {
    "data_vars": "Data variables (:attr:`~xoa.cf.CFSpecs.data_vars`)",
    "coords": "Coordinates (:attr:`~xoa.cf.CFSpecs.coords`)",
    "dims": "Dimensions (:attr:`~xoa.cf.CFSpecs.dims`)",
    }

roles = {
    "data_vars": "cfdatavar",
    "coords": "cfcoord"
    }


def add_items(rst, data):
    for key, value in data.items():
        if key in skip_keys:
            continue
        if isinstance(value, dict):
            rst = add_items(rst, value)
        else:
            if isinstance(value, list):
                value = " ".join(f"``{v!r}``" for v in value)
            else:
                value = f"``{value!r}``"
            rst += f"\t    * - **{key}**\n"
            rst += f"\t      - {value}\n"
        return rst


def genrst(app):

    srcdir = app.env.srcdir

    gendir = os.path.join(srcdir, "gencfspecs")
    if not os.path.exists(gendir):
        os.makedirs(gendir)

    cfspecs = xoa.cf.get_cf_specs()

    rst_tables = {}
    rst_toctrees = {}
    for cfcat in ["data_vars", "coords"]:

        rst_tables[cfcat] = ".. list-table::\n\n"
        rst_toctrees[cfcat] = ".. toctree::\n    :hidden:\n\n"

        logging.info(f"Generating rst files for xoa.cf {cfcat}specs")
        for cfname in cfspecs[cfcat]:

            decdir = os.path.join(gendir, cfcat)
            if not os.path.exists(decdir):
                os.makedirs(decdir)

            # Single declaration file
            role = roles[cfcat]
            with open(os.path.join(decdir, cfname+".rst"), "w") as f:
                title = f':attr:`~xoa.cf.CFSpecs.{cfcat}` [``"{cfname}"``]'
                title += "\n" + len(title) * "="
                rst = title + "\n\n"
                rst += f".. {role}:: {cfname}\n\n"
                rst += "\t.. list-table::\n\n"
                rst = add_items(rst, cfspecs[cfcat][cfname])
                f.write(rst)

            # Append to table
            rst_tables[cfcat] += f"    * - :{role}:`{cfname}`\n"
            rst_tables[cfcat] += "      - {}\n".format(
                cfspecs[cfcat][cfname]["attrs"]["long_name"][0])

            # Append to toctree
            rst_toctrees[cfcat] += f"    gencfspecs/{cfcat}/{cfname}\n"

    rst_tables["dims"] = ".. list-table::\n\n"
    for dim_type, dims in cfspecs["dims"].items():
        rst_tables["dims"] += f"    * - :cfdim:`{dim_type}`\n"
        rst_tables["dims"] += "      - {}\n".format(
            ", ".join([f"``{dim}``" for dim in dims]))

    with open(os.path.join(gendir, "index.txt"), "w") as f:

        for cfcat in rst_tables.keys():

            title = cat_titles[cfcat]
            title = title + "\n" + len(title)*"^"
            f.write(f".. _appendix.cf.{cfcat}:\n\n" + title + "\n\n")
            if cfcat in rst_toctrees:
                f.write(rst_toctrees[cfcat]+"\n")
            f.write(rst_tables[cfcat]+"\n\n")


def setup(app):

    app.add_object_type(
        'cfdatavar', 'cfdatavar',
        objname='xoa.cf CFSpecs.data_vars item',
        indextemplate='pair: %s; xoa.cf CFSpecs.data_vars item')

    app.add_object_type(
        'cfcoord', 'cfcoord',
        objname='xoa.cf CFSpecs.coords item',
        indextemplate='pair: %s; xoa.cf CFSpecs.coords item')

    app.add_object_type(
        'cfdim', 'cfdim',
        objname='xoa.cf CFSpecs.coords.dims item',
        indextemplate='pair: %s; xoa.cf CFSpecs.coords.dims item')

    app.connect('builder-inited', genrst)

    return {'version': '0.1'}
