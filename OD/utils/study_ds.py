import os

import matplotlib.pyplot as plt
import numpy as np
import warnings
from PIL import Image
from matplotlib import patches

ds_dir = r'C:\Users\fsalm\Desktop\DISTILL\LOGIC_firstbatch_163_YOLO\train'
lbls_lst = os.listdir(os.path.join(ds_dir, 'labels'))

neg_samples = 0
pos_samples = 0
w, h = [], []
for f in lbls_lst:
    im = np.array(Image.open(os.path.join(ds_dir, 'images', f[:-3]+'png')))

    boxes = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        boxes = np.loadtxt(os.path.join(ds_dir, 'labels', f), delimiter=" ", dtype=np.float32)

        # fig, ax = plt.subplots()
        # plt.imshow(im)
        # if len(boxes) > 0:
        #     boxes = boxes.reshape(1, -1) if boxes.ndim == 1 else boxes
        #     for c in boxes:
        #         x = c[1] - (c[3] / 2)
        #         y = c[2] - (c[4] / 2)
        #
        #         x *= im.shape[1]
        #         y *= im.shape[0]
        #         w = c[3] * im.shape[1]
        #         h = c[4] * im.shape[0]
        #         rect = patches.Rectangle((x, y), w, h,
        #                                  linewidth=1, edgecolor='r', facecolor='none')
        #         ax.add_patch(rect)
        # plt.show()

    if len(boxes) > 0:
        pos_samples += 1
        boxes = boxes.reshape(1, -1) if boxes.ndim == 1 else boxes
        for bbx in boxes:
            w.append(bbx[3])
            h.append(bbx[4])
    else:
        neg_samples += 1

print(neg_samples, pos_samples)

plt.figure()
plt.hist(w, bins=50)
plt.title('Histogram W')

plt.figure()
plt.hist(h, bins=50)
plt.title('Histogram H')

plt.show()