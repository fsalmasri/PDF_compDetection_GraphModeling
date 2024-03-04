import os

import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import pytesseract
from pytesseract import Output
import cv2
import matplotlib.patches as patches
# from OCR.mmocr.apis import MMOCRInferencer
# from OCR.mmocr.mmocr.apis import MMOCRInferencer

# from OCR.mmocr import mmocr
import easyocr

# import the necessary packages
import numpy as np
from collections import defaultdict
import json

def non_max_suppression_fast(boxes, overlapThresh):
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

reader = easyocr.Reader(['en'], gpu='cuda:0')

f_dir = 'data/LOGIC/0_EFF00_10CRF02.CG'
flst =np.sort(os.listdir(f_dir))
for f in flst:
    print(f)
    img = Image.open(f'{f_dir}/{f}/img.png') #.convert('L')
    img = np.array(img)
    im_h, im_w, _ = img.shape

    result = reader.readtext(f'{f_dir}/{f}/img.png', decoder='greedy',
                             min_size=1, text_threshold=0.2, link_threshold=0.1,
                             height_ths=0, width_ths=0, x_ths=0, y_ths=0)

    # result = reader.detect(f'data/LOGIC/0_EFF00_10CRF02.CG/{f}/img.png',
    #                        min_size=1, text_threshold=0.1, link_threshold=0.1, optimal_num_chars=0,
    #                        height_ths=0, width_ths=0, ycenter_ths=0)



    # bbxs = [[x[0], x[2], x[1], x[3]] for x in result[0][0]]

    bbxs = [[x[0][0][0], x[0][0][1], x[0][2][0], x[0][2][1]] for x in result]

    bbxs, pick = non_max_suppression_fast(np.array(bbxs), overlapThresh=0.9)

    result_selected = [x for idx, x in enumerate(result) if idx in pick]

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
        # print(arr)
        # exit()
        arr.append(txt)
        arr.append(conf)

        dic[idx] = [arr]

    # with open(f"{f_dir}/{f}/OCRbox.json", "w") as jf:
    #     json.dump(dic, jf, indent=4)


    # exit()
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')

    for bb in bbxs:
        rect = patches.Rectangle((bb[0], bb[1]), width=bb[2]-bb[0], height=bb[3]-bb[1], linewidth=1, edgecolor='r',
                                 facecolor='none')
        ax.add_patch(rect)

    plt.show()
    #
