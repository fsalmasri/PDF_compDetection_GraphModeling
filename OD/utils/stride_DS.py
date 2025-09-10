import os
import warnings
import random

import numpy as np
from PIL import Image

from bbxs_utils import convert_xcycwh_xyxy, convert_xyxy_xcycwh

import matplotlib.pyplot as plt
from matplotlib import patches

ds_dir = r'C:\Users\fsalm\Desktop\DISTILL\LOGIC_firstbatch_157_YOLO\valid'
im_lst = os.listdir(os.path.join(ds_dir, 'original_images'))


for im_name in im_lst[:]:
    im = np.array(Image.open(os.path.join(ds_dir, 'original_images', im_name)))

    # print(im_name, im.shape)
    boxes = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        boxes = np.loadtxt(os.path.join(ds_dir, 'original_labels', im_name[:-3]+'txt'), delimiter=" ", dtype=np.float32)

    height, width = im.shape[:2]
    h, w = im.shape[0]/3, im.shape[1]/3
    rh, rw = round(im.shape[0]/3), round(im.shape[1]/3)
    # print(f'Image size: {im.shape}', h, w, round(h), round(w), rh*3, rw*3)

    # print(boxes.shape)
    if len(boxes) > 0:
        coords = convert_xcycwh_xyxy(boxes[:, 1:].copy(), scale=(width, height))
        overlap = 0.6
        crop_size = (1280, 1280)
        crop_h, crop_w = crop_size[0], crop_size[1]
        step_h, step_w = int(crop_h * (1 - overlap)), int(crop_w * (1 - overlap))

        # print(f'Step size: {step_h}')
        counter = 0
        for y in range(0, height, step_h):
            for x in range(0, width, step_w):
                # print(y,x)
                end_y = min(y + crop_h, height)
                end_x = min(x + crop_w, width)
                crop_img = im[y:end_y, x:end_x]

                # shift from let to right
                left_shifted = [0]
                right_shifted = [0]
                up_shifted = [0]
                down_shifted = [0]

                oversize_box = []
                for c_idx, coord in enumerate(coords):
                    xmin, ymin, xmax, ymax = coord[0], coord[1], coord[2], coord[3]
                    if ((xmax - xmin) < (crop_size[0]*0.6)) and ((ymax - ymin) < (crop_size[1]*0.6)):
                        if (xmin < x and xmax < end_x and xmax > x) :
                            left_shifted.append(xmax - x)
                        if (xmin > x and xmin < end_x and xmax > end_x):
                            right_shifted.append(end_x - xmin)

                        if (ymin < y and ymax < end_y and ymax > y) :
                            up_shifted.append(ymax - y)
                        if (ymin > y and ymin < end_y and ymax > end_y):
                            down_shifted.append(end_y - ymin)
                    else:
                        oversize_box.append([boxes[c_idx, 0], xmin, ymin, xmax, ymax])

                max_left_shift = int(max(left_shifted))
                max_righ_shift = int(max(right_shifted))
                max_up_shift = int(max(up_shifted))
                max_down_shift = int(max(down_shifted))

                sum = max_left_shift + max_righ_shift + max_up_shift + max_down_shift
                nx, ny = x, y
                if sum > 0:
                    nx = x + max_left_shift
                    ny = y + max_up_shift
                    end_x -= max_righ_shift
                    end_y -= max_down_shift

                    # print(ny, end_y, nx, end_x)
                    crop_img = im[ny:end_y, nx:end_x]

                new_boxes = []
                for c_idx, coord in enumerate(coords):
                    xmin, ymin, xmax, ymax = coord[0], coord[1], coord[2], coord[3]
                    new_xmin = max(xmin - nx, 0)
                    new_ymin = max(ymin - ny, 0)
                    new_xmax = min(xmax - nx, crop_w)
                    new_ymax = min(ymax - ny, crop_h)

                    if (xmin > nx and xmax < end_x) and (ymin > ny and ymax < end_y):
                        new_boxes.append([boxes[c_idx, 0], new_xmin, new_ymin, new_xmax, new_ymax])
                    else:
                        original_area = (xmax - xmin) * (ymax - ymin)
                        intersection_xmin = max(xmin, nx)
                        intersection_ymin = max(ymin, ny)
                        intersection_xmax = min(xmax, end_x)
                        intersection_ymax = min(ymax, end_y)
                        intersection_area = (max(0, intersection_xmax - intersection_xmin)
                                             * max(0,intersection_ymax - intersection_ymin))

                        # Check if at least 50% of the bounding box is inside the crop
                        if intersection_area >= 0.5 * original_area:
                            new_boxes.append([boxes[c_idx, 0], new_xmin, new_ymin, new_xmax, new_ymax])


                if (crop_img.shape[0] > (crop_size[1] * .4)
                        and crop_img.shape[1] > (crop_size[0] * .4)):
                    crop_im_name = f'{im_name[:-4]}_crop_{counter}.png'
                    crop_im_save = os.path.join(ds_dir, 'images', crop_im_name)
                    crop_lbl_name = f'{im_name[:-4]}_crop_{counter}.txt'
                    crop_lbl_save = os.path.join(ds_dir, 'labels', crop_lbl_name)

                    if len(new_boxes)>0:
                        Image.fromarray(crop_img).save(crop_im_save, quality=100, compression=0)
                        new_boxes = convert_xyxy_xcycwh(np.array(new_boxes), scale=(crop_img.shape[1], crop_img.shape[0]))
                        np.savetxt(crop_lbl_save, new_boxes, newline="\n", fmt="%d %f %f %f %f")
                    else:
                        if random.uniform(0, 1) > 0.7:
                            Image.fromarray(crop_img).save(crop_im_save, quality=100, compression=0)
                            np.savetxt(crop_lbl_save, new_boxes, newline="\n")

                    counter += 1

                new_boxes = convert_xyxy_xcycwh(np.array(new_boxes), scale=(crop_img.shape[1], crop_img.shape[0]))
                print(new_boxes)
                fig, ax = plt.subplots()
                plt.imshow(crop_img)
                for c in new_boxes:
                    x = c[1] - (c[3] / 2)
                    y = c[2] - (c[4] / 2)

                    # w = c[3] - c[1]
                    # h = c[4] - c[2]

                    x *= crop_img.shape[1]
                    y *= crop_img.shape[0]
                    w = c[3] * crop_img.shape[1]
                    h = c[4] * crop_img.shape[0]
                    rect = patches.Rectangle((x, y), w, h,
                                             linewidth=1, edgecolor='r', facecolor='none')
                    ax.add_patch(rect)

                plt.show()
    exit()
