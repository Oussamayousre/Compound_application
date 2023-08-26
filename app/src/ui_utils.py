import base64
import fitz
import io
from PIL import Image, ImageDraw
import random
import matplotlib.pyplot as plt
from datetime import datetime


def get_distinct_colors(n, cmap_name='gist_rainbow'):
    """Returns a list of colors that maps each index in 0, 1, ..., n-1 to a
    distinct RGB color; the keyword argument name must be a standard mpl
    colormap name."""
    rgb_colors = []
    # generate color map from pyplot colormaps
    rgba_color_map = plt.cm.get_cmap(cmap_name, n)
    for i in range(n):
        # get color with i index
        rgba = rgba_color_map(i)
        # convert colors from float space [0,1] to integer space [0, 255].
        r, g, b = int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
        # add color to our list
        rgb_colors.append([r, g, b])
    return rgb_colors


def get_distinct_html_colors(n, cmap_name='gist_rainbow'):
    """Returns a list of colors in HTML code that maps each index in 0, 1, ..., n-1 to a
    distinct RGB color; the keyword argument name must be a standard mpl
    colormap name."""
    # generate distinct RGB colors
    rgb_colors = get_distinct_colors(n, cmap_name)
    html_colors = []
    for color in rgb_colors:
        # convert each color into html color code
        html_colors.append('#{:02x}{:02x}{:02x}'.format(
            color[0], color[1], color[2]))
    return html_colors


