import numpy as np

def convert_xcycwh_xyxy(coords_arr, scale=None):
    '''
    Convert coordinates from XcYcWH format to XYXY format.
    Scale them from normalized values to the image size if image size is provided.
    :param coords_arr: array of coordinates XYWH nX4
    :param scale: (width, height) of image
    :return: coords_arr: array of coordinates XYXY nX4
    '''

    coords_arr[:, 0] = coords_arr[:, 0] - (coords_arr[:, 2] / 2)
    coords_arr[:, 1] = coords_arr[:, 1] - (coords_arr[:, 3] / 2)
    coords_arr[:, 2] = coords_arr[:, 0] + coords_arr[:, 2]
    coords_arr[:, 3] = coords_arr[:, 1] + coords_arr[:, 3]

    if scale is not None:
        width, height = scale[0], scale[1]
        coords_arr[:, 0] *= width
        coords_arr[:, 1] *= height
        coords_arr[:, 2] *= width
        coords_arr[:, 3] *= height

    return coords_arr

def convert_xyxy_xcycwh(coords_arr, scale=None):
    '''
    Convert coordinates from XcYcWH format to XYXY format.
    Scale them from normalized values to the image size if image size is provided.
    :param coords_arr: array of coordinates XYWH nX4
    :param scale: (width, height) of image
    :return: coords_arr: array of coordinates XYXY nX4
    '''

    coords_arr[:, 0] = coords_arr[:, 0].astype(int)

    coords_arr[:, 3] = coords_arr[:, 3] - coords_arr[:, 1]
    coords_arr[:, 4] = coords_arr[:, 4] - coords_arr[:, 2]

    coords_arr[:, 1] = coords_arr[:, 1] + (coords_arr[:, 3] / 2)
    coords_arr[:, 2] = coords_arr[:, 2] + (coords_arr[:, 4] / 2)

    if scale is not None:
        width, height = scale[0], scale[1]
        coords_arr[:, 1] /= width
        coords_arr[:, 2] /= height
        coords_arr[:, 3] /= width
        coords_arr[:, 4] /= height

    return coords_arr
