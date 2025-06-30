import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
from skimage import exposure, color
from matplotlib import colors as mcolors

def multimshow(
    im_pool,
    adjust_thresh,
    chans,
    ch_to_lut,
    nrows,
    ncols,
    show_multi_chan=False,
    text_locs=None,
    texts=None,
    text_colors=None,
    fontsize=10,
    xlabels=None,
    ylabels=None,
    adjust_to_first_im=[True, True],
    mixing_factor=[1, 1],
    order="C",
    merge=True,
    brightness_exclude=None,
):
    # Create figure and axis
    fig, ax = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    plt.subplots_adjust(hspace=0.03, wspace=0.03)
    # fig.set_facecolor("black")

    for a in ax.flatten():
        a.axis("off")

    if nrows == 1:
        ax = np.expand_dims(ax, axis=0)
    if ncols == 1:
        ax = np.expand_dims(ax, axis=1)

    count = 0
    vmin, vmax = [2**16] * len(chans), [0] * len(chans)

    # Calculate exposure limits
    for i, ch in enumerate(chans):
        if not adjust_to_first_im[i]:
            break
        for j, well in enumerate(im_pool.keys()):
            if j in brightness_exclude:
                continue
            im = im_pool[well][ch]
            v1, v2 = np.percentile(im, adjust_thresh[i])
            if v1 < vmin[i]:
                vmin[i] = v1
            if v2 > vmax[i]:
                vmax[i] = v2

    # Create adjusted images
    for j, well in enumerate(im_pool.keys()):
        im_stack = im_pool[well]
        im_rgb = np.dstack([np.zeros_like(im_stack[0])] * 3).astype(np.float64)
        for i, ch in enumerate(chans):
            im = im_stack[ch]
            im_single_chan = np.dstack([np.zeros_like(im)] * 3).astype(np.float64)
            if adjust_to_first_im[i]:
                im_ad = exposure.rescale_intensity(im.astype(np.float64), in_range=(vmin[i], vmax[i])) 
                if im_ad.max() > 1:
                    im_ad = im_ad / vmax[i]
            else:
                v1, v2 = np.percentile(im, adjust_thresh[i])
                im_ad = exposure.rescale_intensity(im.astype(np.float64), in_range=(v1, v2))
                im_ad = im_ad / im_ad.max()
            lut = ch_to_lut[ch]
            if type(lut) is list:
                im_single_chan[:, :, lut] += np.dstack([im_ad] * len(lut))
            else:
                im_single_chan[:, :, lut] += im_ad
            im_rgb += im_single_chan
            if show_multi_chan == "row":
                ax[i, j].imshow(im_single_chan)
        im_rgb = im_rgb / np.max(im_rgb)
        if not show_multi_chan:
            ax.flatten(order=order)[j].imshow(im_rgb, aspect=1)
        else:
            if merge:
                ax[len(chans), j].imshow(im_rgb, aspect=1)
        count += 1

    # Add text
    if texts is not None:
        for loc, text, textcolor in zip(text_locs, texts, text_colors):
            ax[0, 0].text(
                loc[0],
                loc[1],
                text,
                color=textcolor,
                transform=ax[0, 0].transAxes,
                fontsize=fontsize,
            )

    # Add labels
    if ylabels is not None:
        for i, c in enumerate(ylabels):
            ax[i, 0].text(
                -0.05,
                0.5,
                c,
                transform=ax[i, 0].transAxes,
                fontsize=fontsize,
                rotation=90,
                ha="center",
                va="center",
                color="black",
            )
    if xlabels is not None:
        for i, c in enumerate(xlabels):
            ax[0, i].text(
                0.5,
                1.05,
                c,
                color="black",
                transform=ax[0, i].transAxes,
                fontsize=fontsize,
                ha="center",
                va="center",
            )
    return fig, ax



def save_fig(fig_id, path="../figures", tight_layout=True, fmt="pdf", dpi=300):
    Path(path).mkdir(parents=True, exist_ok=True)
    fig_path = Path(path) / f"{fig_id}.{fmt}" 
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    if fmt == "png":
        plt.savefig(fig_path, format=fmt, dpi=dpi)
    else:
        plt.savefig(fig_path, format=fmt, transparent=False)


def defaultStyle(fs=12):
    plt.rc("font", family="Arial")
    plt.rc("text", usetex=False)
    plt.rc("xtick", labelsize=fs)
    plt.rc("ytick", labelsize=fs)
    plt.rc("axes", labelsize=fs)
    plt.rc("mathtext", fontset="custom", rm="Arial")


def create_color_gradient(color1, color2, n_colors):
    # Convert colors to RGB format
    c1_rgb = mcolors.to_rgb(color1)
    c2_rgb = mcolors.to_rgb(color2)

    # Create a linear gradient between the two colors
    mix_pcts = np.linspace(0, 1, n_colors)
    rgb_colors = [
        tuple(a + (b - a) * mix_pct for a, b in zip(c1_rgb, c2_rgb))
        for mix_pct in mix_pcts
    ]

    # Convert RGB to hex format for more consistent behavior
    hex_colors = [mcolors.to_hex(rgb) for rgb in rgb_colors]

    return hex_colors
