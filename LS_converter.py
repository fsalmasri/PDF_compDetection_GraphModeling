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

# class_dict = {0: 'valv', 1: 'compressor', 2: 'car', 3: 'cartoon', 4:'test', 5: 'test1',
#               6: 'test2', 7: 'test3', 8: 'test4', 9: 'test5', 10: 'test6', 11: 'test7',
#               12: 'test8', 13: 'test9',  14: 'test10', 15: 'test11', 16: 'test12', 17: 'test13',
#               -1:'minus'}

class_dict = {x:f'test_{x}' for x in range(-1,350,1)}

for t, v in class_dict.items():
    print(v)
# exit()

Path("LS/annots").mkdir(parents=True, exist_ok=True)
pages_lst = sorted(os.listdir('data'))
for page in pages_lst[:]:

    image_path = f"/data/local-files/?d=images/{page}_cleaned.svg"
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

        if 'class' in v_group:
            cls = [class_dict[v_group['class']]]
        # else:
        #     cls = ['None']

            test = create_rectangle_label_result(k_group, percentage_x, percentage_y, percentage_width, percentage_height,
                                                 cls,
                                                 [image_width, image_height])
            # if not (cls[0] == "test_-1"):
            results.append(test)

    task = create_label_studio_task(1, image_path, results)

    output_file_path = f"LS/annots/{page}.json"
    with open(output_file_path, 'w') as output_file:
        json.dump(task, output_file, indent=4)


