"""Generate files to declare and list default CF specifications"""

import os
import logging

import xoa.cf


skip_keys = ["inherit", "processed"]

cat_titles = {
    "data_vars": "Data variables (:attr:`~xoa.cf.CFSpecs.data_vars`)",
    "coords": "Coordinates (:attr:`~xoa.cf.CFSpecs.coords`)",
    "dims": "Dimensions (:attr:`~xoa.cf.CFSpecs.dims`)",
    'sections': 'Other sections',
    }

roles = {
    "data_vars": "cfdatavar",
    "coords": "cfcoord"
    }


def add_items(rst, data, indent=0, subformat='[`{key}`]'):
    local = ""
    nest = {}
    for key, value in data.items():
        if key in skip_keys:
            continue
        if isinstance(value, dict):
            nest[key] = value
        else:
            if isinstance(value, list):
                value = " ".join(f"``{v!r}``" for v in value)
            else:
                value = f"``{value!r}``"
            local += "\t"*indent + f"    * - **{key}**\n"
            local += "\t"*indent + f"      - {value}\n"
    if local:
        rst += "\t"*indent + ".. list-table::\n\n" + local

    for key, dvalue in nest.items():
        rst += "\n" + "\t"*indent + f'.. rubric:: [``"{key}"``]\n\n'
        if dvalue:
            rst = add_items(rst, dvalue, indent=indent)
        else:
            rst += "\t"*indent + "Empty dictionary\n\n"
    return rst


def genrst(app):

    srcdir = app.env.srcdir

    gendir = os.path.join(srcdir, "gencfspecs")
    if not os.path.exists(gendir):
        os.makedirs(gendir)

    cfspecs = xoa.cf.get_cf_specs()
    comments = xoa.cf._get_cfgm_().specs.inline_comments

    rst_tables = {}
    rst_toctrees = {}

    # Data vars and coordinates
    for cfcat in ["data_vars", "coords"]:

        rst_tables[cfcat] = ".. list-table::\n\n"
        rst_toctrees[cfcat] = ".. toctree::\n    :hidden:\n\n"

        logging.info(f"Generating rst files for xoa.cf {cfcat} specs")
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
                rst = add_items(rst, cfspecs[cfcat][cfname], indent=1)
                f.write(rst)

            # Append to table
            rst_tables[cfcat] += f"    * - :{role}:`{cfname}`\n"
            rst_tables[cfcat] += "      - {}\n".format(
                cfspecs[cfcat][cfname]["attrs"]["long_name"][0])

            # Append to toctree
            rst_toctrees[cfcat] += f"    gencfspecs/{cfcat}/{cfname}\n"

    # Dimensions
    rst_tables["dims"] = ".. list-table::\n\n"
    for dim_type, dims in cfspecs["dims"].items():
        rst_tables["dims"] += f"    * - :cfdim:`{dim_type}`\n"
        rst_tables["dims"] += "      - {}\n".format(
            ", ".join([f"``{dim}``" for dim in dims]))

    # Other sections
    rst_tables["sections"] = ".. list-table::\n\n"
    rst_toctrees["sections"] = ".. toctree::\n    :hidden:\n\n"
    for section in "register", "sglocator", "vertical", "accessors":

        logging.info(f"Generating rst files for xoa.cf {section} specs")
        with open(os.path.join(gendir, section+".rst"), "w") as f:
            title = f':attr:`~xoa.cf.CFSpecs` [``"{section}"``]'
            title += "\n" + len(title) * "="
            rst = title + "\n\n"
            rst += f".. cfsec:: {section}\n\n"
            rst = add_items(rst, cfspecs[section], indent=1)
            f.write(rst)

        # Append to table
        rst_tables["sections"] += f"    * - :cfsec:`{section}`\n"
        rst_tables["sections"] += "      - {}\n".format(
            comments[section].strip("# "))

        # Append to toctree
        rst_toctrees["sections"] += f"    gencfspecs/{section}\n"

    # Write the index.txt
    with open(os.path.join(gendir, "index.txt"), "w") as f:

        for key, title in cat_titles.items():

            # Title
            title = title + "\n" + len(title)*"^"
            f.write(f".. _appendix.cf.{key}:\n\n" + title + "\n\n")

            # Hidden toctree
            if key in rst_toctrees:
                f.write(rst_toctrees[key]+"\n")

            # Table
            f.write(rst_tables[key]+"\n\n")



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

    app.add_object_type(
        'cfsec', 'cfsec',
        objname='xoa.cf CFSpecs section',
        indextemplate='pair: %s; xoa.cf CFSpec section')

    app.connect('builder-inited', genrst)

    return {'version': '0.1'}
