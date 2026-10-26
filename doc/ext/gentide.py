"""Generate the table of the default tidal constituents"""

import os
import logging

import numpy as np

import xoa.tide


def genrst(app):
    gendir = os.path.join(app.env.srcdir, "gentide")
    os.makedirs(gendir, exist_ok=True)
    logging.info("Generating the table of tidal constituents")

    db = xoa.tide.ConstituentDB(xoa.tide.data.CFG_FILE)  # the default file only
    names = db.names
    with xoa.tide.set_constituents(db, replace=True):
        periods = 2 * np.pi / xoa.tide.get_frequencies(names) / 3600
    rst = ".. list-table::\n    :header-rows: 1\n\n"
    rst += "    * - Name\n      - Species\n      - Doodson number\n"
    rst += "      - Period (h)\n      - Satellites\n      - Compound of\n"
    for name, period in zip(names, periods):
        const = db[name]
        compound = " + ".join(
            f"{mult}×{src}" if mult > 1 else src for src, mult in const["compound_of"] or []
        )
        rst += f"    * - ``{name}``\n      - {const['species']}\n"
        rst += f"      - ``{const['doodson']}``\n      - {period:.3f}\n"
        rst += f"      - {len(const['satellites'])}\n      - {compound}\n"
    with open(os.path.join(gendir, "constituents.txt"), "w") as f:
        f.write(rst)


def setup(app):
    app.connect("builder-inited", genrst)
    return {"parallel_read_safe": True}
