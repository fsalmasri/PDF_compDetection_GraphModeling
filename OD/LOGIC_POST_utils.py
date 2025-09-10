import numpy as np
import random
from collections import Counter

from shapely.geometry import Polygon


def find_first_zero(rows):
    first_zero_indices = []
    for row in rows:
        try:
            first_zero_index = np.where(row == 0)[0][0]
        except IndexError:
            first_zero_index = None  # If no zero is found in the row
        first_zero_indices.append(first_zero_index)

    filtered_positions = [index for index in first_zero_indices if index is not None]

    index_counter = Counter(filtered_positions)
    return index_counter.most_common(1)[0][0]


def find_symbols_by_profile_projection(im_batch):
    h, w = im_batch.shape
    row_projections = random.sample(range(0, h), int(h*.3))
    column_projections = random.sample(range(0, w), int(w * .3))

    im_batch_rows = im_batch[row_projections, :]
    im_batch_columns = im_batch[:, column_projections].T

    row_findex = find_first_zero(im_batch_rows)
    cl_findex = find_first_zero(im_batch_columns)

    flipped_im_batch_rows = np.fliplr(im_batch_rows)
    row_lindex = w - find_first_zero(flipped_im_batch_rows)
    flipped_im_batch_columns = np.fliplr(im_batch_columns)
    cl_lindex = h - find_first_zero(flipped_im_batch_columns) - 0

    return row_findex, cl_findex, row_lindex, cl_lindex


def find_LC_coords(bbx, im):

    tim = im.copy().mean(axis=2)
    tim[tim!=255] = 0

    x0 = int(bbx[0])
    y0 = int(bbx[1])
    xn = int(bbx[2])
    yn = int(bbx[3])


    im_batch = tim[y0:yn, x0:xn]

    px0, py0, pxn, pyn = find_symbols_by_profile_projection(im_batch)

    px0 += x0
    pxn += x0
    py0 += y0
    pyn += y0

    return px0, py0, pxn, pyn

def get_Polygon(coords):
    x0, y0, xn, yn = coords
    cord0 = (x0, y0)
    cord1 = (xn, y0)
    cord2 = (xn, yn)
    cord3 = (x0, yn)

    return Polygon([cord0, cord1, cord2, cord3])