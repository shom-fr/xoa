#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
To generate the xoa logos
"""
import os
import logging

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms

shomlightblue = (90, 194, 231)
shomdarkblue = (0, 36, 84)


def genlogo(outfile, dark=False):
    """Generate a xoa logo and save it"""
    font = "Cantarell"
    # font = "Noto sans"
    width, height = (4, 2.7)

    if dark:
        fontcolor = "w"
    else:
        fontcolor = tuple(c / 255 for c in shomdarkblue)
    circlecolor = tuple(c / 255 for c in shomlightblue)

    with plt.rc_context({"font.sans-serif": [font]}):
        fig = plt.figure(figsize=(width, height))
        if not dark:
            fig.patch.set_facecolor("w")
        ax = plt.axes([0, 0, 1, 1], aspect=1, facecolor="b")
        kw = dict(
            family="sans-serif",
            size=100,
            color=fontcolor,
            va="center_baseline",
            weight="extra bold",
            transform=ax.transAxes,
        )
        ax.text(0.05, 0.515, "X", ha="left", **kw)
        ax.text(0.95, 0.515, "A", ha="right", **kw)
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)

        clip_height = 0.26
        for y0 in (0, 1 - clip_height):
            circle = mpatches.Circle(
                (width / 2, height / 2),
                radius=height * 0.5 * 0.81,
                facecolor="none",
                linewidth=14,
                ec=circlecolor,
            )

            ax.add_patch(circle)

            clip = mtransforms.TransformedBbox(
                mtransforms.Bbox([[0, y0], [1, y0 + clip_height]]), ax.transAxes
            )
            circle.set_clip_box(clip)

        ax.axis("off")
        fig.savefig(outfile, transparent=dark)
        plt.close(fig)
        del fig


def genlogos(app):
    """Generate light and dark xoa logo during doc compilation"""
    srcdir = app.env.srcdir
    gendir = os.path.join(srcdir, "_static")

    logging.debug("Generating light xoa logo...")
    genlogo(os.path.join(gendir, "xoa-logo-light.png"))
    logging.info("Generated light xoa logo")

    logging.debug("Generating dark xoa logo...")
    genlogo(os.path.join(gendir, "xoa-logo-dark.png"), dark=True)
    logging.info("Generated dark xoa logo")


def setup(app):
    app.connect('builder-inited', genlogos)
    return {'version': '0.1'}


if __name__ == "__main__":
    genlogo("xoa-logo-light.png")
    genlogo("xoa-logo-dark.png", dark=True)
