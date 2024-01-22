import json
import os
import shutil
from pathlib import Path

from extractor.utils import  keystoint

def create_label_studio_task(id, image_path, results):
    task = {
        "id": id,
        "data": {
            "image": image_path
        },
        "annotations":[
            {
                "result": results
            }
        ]
    }
    return task


def create_rectangle_label_result(id, x, y, width, height, labels, img_size):
    return {
          "original_width": img_size[0],
          "original_height": img_size[1],
          "image_rotation": 0,
          "value": {
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "rotation": 0,
            "rectanglelabels": labels
          },
          "id": id,
          "from_name": "label",
          "to_name": "image",
          "type": "rectanglelabels",
          "origin": "manual"
        }


# image_path = "/data/upload/1/f5099cd7-0.png"
# image_path = "/data/local-files?d=images/0.png"
image_path = "/data/local-files/?d=images/0.svg"


pages_lst = os.listdir('data')
for page in pages_lst[:1]:
    grouped_prims = json.load(open(f'data/{page}/grouped_prims.json'), object_hook=keystoint)
    page_info = json.load(open(f'data/{page}/info.json'), object_hook=keystoint)

    image_width = page_info['width']
    image_height = page_info['height']

    results = []
    for k_group, v_group in grouped_prims.items():
        x, y, width, height = v_group['bbx']

        # Convert to percentages
        percentage_x = (x / image_width) * 100
        percentage_y = (y / image_height) * 100
        percentage_width = (width / image_width) * 100
        percentage_height = (height / image_height) * 100

        test = create_rectangle_label_result(k_group, percentage_x, percentage_y, percentage_width, percentage_height,
                                             ["valve"], [image_width, image_height])
        results.append(test)

    task = create_label_studio_task(1, image_path, results)
    output_file_path = f"LS/annots/{page}.json"
    with open(output_file_path, 'w') as output_file:
        json.dump(task, output_file, indent=4)


