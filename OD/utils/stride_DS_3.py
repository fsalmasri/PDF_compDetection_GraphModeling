import os
import warnings
import random
import pathlib

import numpy as np
from PIL import Image

from bbxs_utils import convert_xcycwh_xyxy, convert_xyxy_xcycwh
from strides_utils import save_crop_bbx, plot_crop_bbx, check_intersection_area, check_if_bbx_one_corner_in_crop

version = 7

package = 'train' # 'valid' #
src_dir = f'../../../LOGICS/LOGIC 295_v2/Distill_logics_splitted/{package}'
dst_dir = f'../../../LOGICS/LOGIC 295_v2/Distill_logics_cropped_{version}/{package}'

im_lst = os.listdir(os.path.join(src_dir, 'images'))
print(f'found {len(im_lst)} images')

pathlib.Path(os.path.join(dst_dir, 'images')).mkdir(parents=True, exist_ok=True)
pathlib.Path(os.path.join(dst_dir, 'labels')).mkdir(parents=True, exist_ok=True)


empty_counter = 0
for im_name in im_lst[:]:
    im = np.array(Image.open(os.path.join(src_dir, 'images', im_name)))


    boxes = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        boxes = np.loadtxt(os.path.join(src_dir, 'labels', im_name[:-3] + 'txt'), delimiter=" ", dtype=np.float32)

    height, width = im.shape[:2]

    # print(boxes.shape)
    if len(boxes) > 0:
        coords = convert_xcycwh_xyxy(boxes[:, 1:].copy(), scale=(width, height))

        overlap = 0.98 #0.7

        # crop_size = (1280, 1280)
        # name_counter = 0
        crop_size = (2048, 2048)
        name_counter = 100

        crop_h, crop_w = crop_size[0], crop_size[1]
        step_h, step_w = int(crop_h * (1 - overlap)), int(crop_w * (1 - overlap))

        print(im_name, im.shape, f'Step size: {step_h}')

        for y in range(0, height, step_h):
            for x in range(0, width, step_w):
                end_y = min(y + crop_h, height)
                end_x = min(x + crop_w, width)
                crop_img = im[y:end_y, x:end_x]

                new_boxes = []
                gpart_flag = False
                for c_idx, coord in enumerate(coords):
                    xmin, ymin, xmax, ymax = coord[0], coord[1], coord[2], coord[3]

                    # check if bbx within the new crop
                    # in_crop = check_if_bbx_in_crop(x, y, end_x, end_y, xmin, ymin, xmax, ymax)
                    intersect, part_flag = check_intersection_area(x, y, end_x, end_y, xmin, ymin, xmax, ymax, threshold=1)

                    if part_flag:
                        gpart_flag = True
                        break

                    if intersect:
                        # If bbx intersects then check if one of its corner inside the crop. otherwise break the entire loop.
                        if not check_if_bbx_one_corner_in_crop(x, y, end_x, end_y, xmin, ymin, xmax, ymax):
                            gpart_flag = True
                            break

                        new_xmin = max(xmin - x, 0)
                        new_ymin = max(ymin - y, 0)
                        new_xmax = min(xmax - x, crop_w)
                        new_ymax = min(ymax - y, crop_h)

                        margin = 2
                        new_xmin -= margin
                        new_ymin -= margin
                        new_xmax += margin
                        new_ymax += margin

                        new_boxes.append([boxes[c_idx, 0], new_xmin, new_ymin, new_xmax, new_ymax])

                if not gpart_flag and len(new_boxes) > 0:

                    #Check big LC and save them for augmentation:
                    big_lc_flag = False
                    for n in new_boxes:
                        if n[0] == 0:
                            if (n[4] - n[2]) / 2048 > 0.5:
                                big_lc_flag = True


                    if big_lc_flag:
                        # This is for saving
                        n_counter, e_counter = save_crop_bbx(dst_dir, 'images', 'labels',
                                                             crop_size, im_name, name_counter, crop_img, new_boxes,
                                                             empty_th=.98)
                        if n_counter: name_counter += 1
                        if e_counter: empty_counter += 1


                        # This is for plotting only
                        # plot_crop_bbx(new_boxes, crop_img)

print(empty_counter)