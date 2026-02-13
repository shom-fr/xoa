"""Generate files to declare and list default meta specifications"""

import os
import logging

import xoa.meta
import xoa.meta.configs


skip_keys = ["inherit", "processed"]

cat_titles = {
    "data_vars": "Data variables (:attr:`~xoa.meta.MetaSpecs.data_vars`)",
    "coords": "Coordinates (:attr:`~xoa.meta.MetaSpecs.coords`)",
    "dims": "Dimensions (:attr:`~xoa.meta.MetaSpecs.dims`)",
    'sections': 'Other sections',
}

roles = {"data_vars": "metadatavar", "coords": "metacoord"}


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
            local += "\t" * indent + f"    * - **{key}**\n"
            local += "\t" * indent + f"      - {value}\n"
    if local:
        rst += "\t" * indent + ".. list-table::\n\n" + local

    for key, dvalue in nest.items():
        rst += "\n" + "\t" * indent + f'.. rubric:: [``"{key}"``]\n\n'
        if dvalue:
            rst = add_items(rst, dvalue, indent=indent)
        else:
            rst += "\t" * indent + "Empty dictionary\n\n"
    return rst


def genrst(app):

    srcdir = app.env.srcdir

    gendir = os.path.join(srcdir, "genmetaspecs")
    if not os.path.exists(gendir):
        os.makedirs(gendir)

    meta_specs = xoa.meta.get_meta_specs()
    comments = xoa.meta.general._get_cfgm_().specs.inline_comments

    rst_tables = {}
    rst_toctrees = {}

    # Data vars and coordinates
    for meta_cat in ["data_vars", "coords"]:

        rst_tables[meta_cat] = ".. list-table::\n\n"
        rst_toctrees[meta_cat] = ".. toctree::\n    :hidden:\n\n"

        logging.info(f"Generating rst files for xoa.meta {meta_cat} specs")
        for meta_name in meta_specs[meta_cat]:

            decdir = os.path.join(gendir, meta_cat)
            if not os.path.exists(decdir):
                os.makedirs(decdir)

            # Single declaration file
            role = roles[meta_cat]
            with open(os.path.join(decdir, meta_name + ".rst"), "w") as f:
                title = f':attr:`~xoa.meta.MetaSpecs.{meta_cat}` [``"{meta_name}"``]'
                title += "\n" + len(title) * "="
                rst = title + "\n\n"
                rst += f".. {role}:: {meta_name}\n\n"
                rst = add_items(rst, meta_specs[meta_cat][meta_name], indent=1)
                f.write(rst)

            # Append to table
            rst_tables[meta_cat] += f"    * - :{role}:`{meta_name}`\n"
            rst_tables[meta_cat] += "      - {}\n".format(
                meta_specs[meta_cat][meta_name]["attrs"]["long_name"][0]
            )

            # Append to toctree
            rst_toctrees[meta_cat] += f"    genmetaspecs/{meta_cat}/{meta_name}\n"

    # Dimensions
    rst_tables["dims"] = ".. list-table::\n\n"
    for dim_type, dims in meta_specs["dims"].items():
        rst_tables["dims"] += f"    * - :metadim:`{dim_type}`\n"
        rst_tables["dims"] += "      - {}\n".format(", ".join([f"``{dim}``" for dim in dims]))

    # Other sections
    rst_tables["sections"] = ".. list-table::\n\n"
    rst_toctrees["sections"] = ".. toctree::\n    :hidden:\n\n"
    for section in "register", "sglocator", "vertical", "accessors":

        logging.info(f"Generating rst files for xoa.meta {section} specs")
        with open(os.path.join(gendir, section + ".rst"), "w") as f:
            title = f':class:`~xoa.meta.MetaSpecs` [``"{section}"``]'
            title += "\n" + len(title) * "="
            rst = title + "\n\n"
            rst += f".. metasec:: {section}\n\n"
            rst = add_items(rst, meta_specs[section], indent=1)
            f.write(rst)

        # Append to table
        rst_tables["sections"] += f"    * - :metasec:`{section}`\n"
        rst_tables["sections"] += "      - {}\n".format(comments[section].strip("# "))

        # Append to toctree
        rst_toctrees["sections"] += f"    genmetaspecs/{section}\n"

    # Write the index.txt
    with open(os.path.join(gendir, "index.txt"), "w") as f:

        for key, title in cat_titles.items():

            # Title
            title = title + "\n" + len(title) * "^"
            f.write(f".. _appendix.meta.{key}:\n\n" + title + "\n\n")

            # Hidden toctree
            if key in rst_toctrees:
                f.write(rst_toctrees[key] + "\n")

            # Table
            f.write(rst_tables[key] + "\n\n")

    # Specialized meta specs
    with open(os.path.join(gendir, "specialized.txt"), "w") as fi:
        fi.write(".. toctree::\n\n")
        for name in list(xoa.meta.configs.META_CONFIGS):
            logging.info(f"Generating rst file for xoa.meta.configs.META_CONFIGS[{name}]")
            title = f"Meta specifications for ``{name}`` datasets"
            title = title + "\n" + "=" * len(title)
            with open(os.path.join(gendir, name + ".rst"), "w") as f:
                f.write(f".. _appendix.meta.specialized.{name}:\n\n{title}\n\n")
                f.write(
                    f".. literalinclude:: ../../xoa/meta/configs/{name}.cfg\n    :language: ini\n\n"
                )
            fi.write(f"    genmetaspecs/{name}\n")
        fi.write("\n")


def setup(app):

    app.add_object_type(
        'metadatavar',
        'metadatavar',
        objname='xoa.meta MetaSpecs.data_vars item',
        indextemplate='pair: %s; xoa.meta MetaSpecs.data_vars item',
    )

    app.add_object_type(
        'metacoord',
        'metacoord',
        objname='xoa.meta MetaSpecs.coords item',
        indextemplate='pair: %s; xoa.meta MetaSpecs.coords item',
    )

    app.add_object_type(
        'metadim',
        'metadim',
        objname='xoa.meta MetaSpecs.coords.dims item',
        indextemplate='pair: %s; xoa.meta MetaSpecs.coords.dims item',
    )

    app.add_object_type(
        'metasec',
        'metasec',
        objname='xoa.meta MetaSpecs section',
        indextemplate='pair: %s; xoa.meta MetaSpec section',
    )
    app.add_object_type(
        'metaopt',
        'metaopt',
        objname='xoa.meta MetaSpecs option',
        indextemplate='pair: %s; xoa.meta MetaSpec option',
    )

    app.connect('builder-inited', genrst)

    return {'version': '0.1'}
