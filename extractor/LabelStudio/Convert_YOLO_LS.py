import json
import os
from LS_converter import create_rectangle_label_result, create_label_studio_task
from extractor.utils import adjust_bbx_4coords_xywh


# This function convert the annotated labels in Label-studio into a new label-studio Json file.
# the reason for this function is to add a new automatic labeled classes. Here we added the LC connectors.

f = open('data/project-4-at-2024-06-12-12-39-6d83d116.json')
minijson = json.load(f)

def load_grouped_prims(image_path):
    folder_page_name = image_path.split('/')[-1]
    folder_page_splits = folder_page_name.split('_p_')
    folder_name = folder_page_splits[0]
    page = folder_page_splits[-1][:-4]

    f = open(os.path.join('../../data/LOGIC', folder_name, page, 'grouped_prims.json'))
    grouped_primes = json.load(f)

    f = open(os.path.join('../../data/LOGIC', folder_name, page, 'info.json'))
    info = json.load(f)

    return grouped_primes, [info['width'], info['height']]


for m_idx, m in enumerate(minijson):
    if 'label' in m:
        image_path = m['image']


        lables = m['label']
        results = []
        for l_idx, l in enumerate(lables):

            x = l['x']
            y = l['y']
            w = l['width']
            h = l['height']

            image_width = l['original_width']
            image_height = l['original_height']
            rec_labels = l['rectanglelabels']

            rectangle_label = create_rectangle_label_result(l_idx, [x, y, w, h],
                                                            rec_labels, [image_width, image_height])
            results.append(rectangle_label)

        LC_primes_K_max = len(lables) + 1


        grouped_primes, [image_width, image_height] = load_grouped_prims(image_path)
        con_primes = {k_g: v_g for k_g, v_g in grouped_primes.items() if v_g['cls'] == 'LC_con'}

        for k_group, v_group in con_primes.items():
            x, y, w, h = adjust_bbx_4coords_xywh(v_group['bbx'], margin=3)

            # Convert to percentages
            percentage_x = (x / image_width) * 100
            percentage_y = (y / image_height) * 100
            percentage_width = (w / image_width) * 100
            percentage_height = (h / image_height) * 100
            coords = [percentage_x, percentage_y, percentage_width, percentage_height]

            rectangle_label = create_rectangle_label_result(int(k_group)+int(LC_primes_K_max),
                                                            coords,
                                                            ['LC_CON'],
                                                            [image_width, image_height])

            results.append(rectangle_label)

        task = create_label_studio_task(m_idx, image_path, results)

        # output_file_path = f"{folder_to_save_annots}/{file_to_save}.json"
        file_name_save = image_path.split('/')[-1][:-3]
        output_file_path = f'../../LS/LOGIC/annots/{file_name_save}.json'
        with open(output_file_path, 'w') as output_file:
            json.dump(task, output_file, indent=4)

        # print(task)
        # exit()
