from PIL import Image
import easyocr

import numpy as np
from collections import defaultdict
import json

def non_max_suppression_fast(boxes, overlapThresh):
    '''
    Remove overlapped bounding boxes using NMS algorithm
    :param boxes: list of bbxs
    :param overlapThresh: overlap threshold
    :return: list of bbxs.
    '''
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    # initialize the list of picked indexes
    pick = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))
    # return only the bounding boxes that were picked using the
    # integer data type

    return boxes[pick].astype("int"), pick

def run_OCR(fpath, reader):
    '''
    This function run easyocr modle to detect all text present in the drawing image, extract bounding boxes,
    remove overlapped bounding boxes.
    :param fpath: path to drawing image
    :return: a dictorinary with index key and value of [bbx_x0, bbx_x1, bbx_y0, bbx_y1, text, confidence]
    '''


    img = Image.open(f'{fpath}/img.png')  # .convert('L')
    img = np.array(img)
    im_h, im_w, _ = img.shape

    # result = reader.detect(f'data/LOGIC/0_EFF00_10CRF02.CG/{f}/img.png',
    #                        min_size=1, text_threshold=0.1, link_threshold=0.1, optimal_num_chars=0,
    #                        height_ths=0, width_ths=0, ycenter_ths=0)

    result = reader.readtext(f'{fpath}/img.png', decoder='greedy', mag_ratio=1.5,
                              min_size=1, text_threshold=0.2, low_text=0.3,
                              width_ths=0.4, height_ths=.3)
                             #   link_threshold=0.1,
                             #  x_ths=0, y_ths=0)

    # bbxs = [[x[0], x[2], x[1], x[3]] for x in result[0][0]]

    bbxs = [[x[0][0][0], x[0][0][1], x[0][2][0], x[0][2][1]] for x in result]

    bbxs, pick = non_max_suppression_fast(np.array(bbxs), overlapThresh=0.9)

    result_selected = result # [x for idx, x in enumerate(result) if idx in pick]

    dic = defaultdict(list)
    for idx, res in enumerate(result_selected):
        bbx = res[0]
        txt = res[1]
        conf = float(res[2])

        arr = []

        arr.append(float(bbx[0][0] / im_w))
        arr.append(float(bbx[0][1] / im_h))
        arr.append(float(bbx[1][0] / im_w))
        arr.append(float(bbx[2][1] / im_h))

        arr.append(txt)
        arr.append(conf)

        dic[idx] = [arr]

    return dic, bbxs

def run_paddle_OCR(fpath, reader):
    img = Image.open(f'{fpath}/img.png')  # .convert('L')
    img = np.array(img)
    im_h, im_w, _ = img.shape

    result = reader.ocr(f'{fpath}/img.png', cls=True)

    result = result[0]
    bbxs = [[x[0][0][0], x[0][0][1], x[0][2][0], x[0][2][1]] for x in result]

    bbxs, pick = non_max_suppression_fast(np.array(bbxs), overlapThresh=0.9)

    result_selected = result #[x for idx, x in enumerate(result) if idx in pick]

    dic = defaultdict(list)
    for idx, res in enumerate(result_selected):
        bbx = res[0]
        txt = res[1][0]
        conf = float(res[1][1])

        arr = []

        arr.append(float(bbx[0][0] / im_w))
        arr.append(float(bbx[0][1] / im_h))
        arr.append(float(bbx[1][0] / im_w))
        arr.append(float(bbx[2][1] / im_h))

        arr.append(txt)
        arr.append(conf)

        dic[idx] = [arr]

    return dic, bbxs

def save_OCR_bbx(fpath, cuda=True):

    if cuda:
        reader = easyocr.Reader(['en'], gpu='cuda:0')
    else:
        reader = easyocr.Reader(['en'], gpu=False)

    dic, _ = run_OCR(fpath, reader)

    with open(f"{fpath}/OCRbox.json", "w") as jf:
        json.dump(dic, jf, indent=4)
