import json
import os
import shutil
from pathlib import Path

from extractor.utils import  keystoint

def xywh_to_yolo(image_width, image_height, x, y, width, height):
    x_center = (x + width / 2) / image_width
    y_center = (y + height / 2) / image_height
    normalized_width = width / image_width
    normalized_height = height / image_height

    return x_center, y_center, normalized_width, normalized_height


# Prepare Label Studio annots folder
Path("annots/images").mkdir(parents=True, exist_ok=True)
Path("annots/labels").mkdir(parents=True, exist_ok=True)

file_path = "annots/classes.txt"
data_lines = ['class1', 'class2', 'class3']

with open(file_path, 'w') as file:
    for line in data_lines:
        file.write(line + '\n')


pages_lst = os.listdir('data')
for page in pages_lst[:1]:
    grouped_prims = json.load(open(f'data/{page}/grouped_prims.json'), object_hook=keystoint)
    page_info = json.load(open(f'data/{page}/info.json'), object_hook=keystoint)

    image_width = page_info['width']
    image_height = page_info['height']

    # //TODO change file name to random
    shutil.copyfile(f'data/{page}/{page}.png', f'annots/images/{page}.png')

    bbx_fpath = f'annots/labels/{page}.txt'
    with open(bbx_fpath, 'a') as file:
        for k_group, v_group in grouped_prims.items():
            x, y, width, height = v_group['bbx']

            yolo_format = xywh_to_yolo(image_width, image_height, x, y, width, height)
            # print(f'Object: {k_group}, YOLO Format: {yolo_format}')

            line = f"0 {yolo_format[0]} {yolo_format[1]} {yolo_format[2]} {yolo_format[3]}\n"
            file.write(line)

