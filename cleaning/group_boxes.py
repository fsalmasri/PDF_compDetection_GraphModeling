import json
import os

import numpy as np

from extractor.utils import keystoint
from collections import defaultdict

import networkx as nx

flst = sorted(os.listdir('../LS/annots'))
labels_lst = set()
# labels_to_group = {'test_30', 'test_3', 'test_19', 'test_18'} #1052
labels_to_group = {'test_15', 'test_17', 'test_16', '1057', 'test_29', 'test_40', 'test_41'} #1057


def overlap(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    intersection_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    intersection_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))

    area_intersection = intersection_x * intersection_y
    area_box1 = w1 * h1
    area_box2 = w2 * h2

    iou = area_intersection / float(area_box1 + area_box2 - area_intersection)
    return iou > 0.1



def check_boxes(boxes, secondcls_boxes):

    relations = []
    for symbol in boxes:
        bbx1_idx = symbol['idx']
        bbx1_k_id = symbol['k_id']
        bbx1 = symbol['bbx']

        # print(bbx1_idx, bbx1_k_id, bbx1)

        for seccls in secondcls_boxes:
            sec_boxes = secondcls_boxes[seccls]

            for symbol in sec_boxes:
                bbx2_idx = symbol['idx']
                bbx2_k_id = symbol['k_id']
                bbx2 = symbol['bbx']

                # print(bbx2_idx, bbx2_k_id, bbx2)

                if overlap(bbx1, bbx2):
                    relations.append([bbx1_idx, bbx2_idx, bbx1_k_id, bbx2_k_id])

    return relations

def detect_overlap(detected_boxes):

    # print(detected_boxes)

    relations = []
    keys = list(detected_boxes.keys())
    for i in range(len(keys)):
        boxes = detected_boxes[keys[i]]
        secondcls_boxes = {x:v for x,v in detected_boxes.items() if x != keys[i]}
        relations.extend(check_boxes(boxes, secondcls_boxes))

    return relations


def merge_bbx(bbx_to_merge):
    #convert w, h to x y
    for bbx in bbx_to_merge:
        bbx[2] = bbx[0] + bbx[2]
        bbx[3] = bbx[1] + bbx[3]

    min_x = np.min(np.array(bbx_to_merge)[:, 0])
    min_y = np.min(np.array(bbx_to_merge)[:, 1])
    max_x = np.max(np.array(bbx_to_merge)[:, 2])
    max_y = np.max(np.array(bbx_to_merge)[:, 3])

    merged_box = [min_x, min_y, max_x-min_x, max_y-min_y]

    return merged_box


for f in flst:
    with open(f'../LS/annots/{f}', 'r') as jf:
        annot = json.load(jf)

    with open(f'../data/{f[:-5]}/grouped_prims.json', 'r') as jf:
        gprims = json.load(jf, object_hook=keystoint)

    results_lst = annot['annotations'][0]['result']


    detected_boxes = defaultdict(list)
    for idx, res in enumerate(results_lst):

        label = res['value']['rectanglelabels'][0]
        value = res['value']
        k_id = res['id']

        if label in labels_to_group:
            current_box = {'idx': idx, 'k_id': k_id,
                           'bbx': [value['x'], value['y'], value['width'], value['height']]}
            detected_boxes[label].append(current_box)

    if len(detected_boxes) > 1:
        print(f)
        relations = detect_overlap(detected_boxes)

        # Add relations to graph to find connected comp.
        G = nx.Graph()
        [G.add_edge(rel[0], rel[1]) for rel in relations]
        connected_symbols = list(nx.connected_components(G))

        # print(relations)
        # print(connected_symbols)

        symbols_to_add = []

        for con_symbol in connected_symbols:
            bbx_result_to_merge = [x for idx, x in enumerate(results_lst) if idx in con_symbol]
            bbx_to_merge = [[x['value']['x'], x['value']['y'], x['value']['width'], x['value']['height']] for x in bbx_result_to_merge]
            merged_box = merge_bbx(bbx_to_merge)

            merged_result = bbx_result_to_merge[0]
            merged_result['value']['x'] = merged_box[0]
            merged_result['value']['y'] = merged_box[1]
            merged_result['value']['width'] = merged_box[2]
            merged_result['value']['height'] = merged_box[3]
            merged_result['value']['rectanglelabels'] = ['1057']

            symbols_to_add.append(merged_result)

        print(len(results_lst))
        flattened_connected_symbols = [element for subset in connected_symbols for element in subset]
        results_lst= [x for idx, x in enumerate(results_lst) if idx not in flattened_connected_symbols]
        print(len(results_lst))

        [results_lst.append(x) for x in symbols_to_add]
        annot['annotations'][0]['result'] = results_lst

    output_file_path = f"../LS/annots/{f}"
    with open(output_file_path, 'w') as output_file:
        json.dump(annot, output_file, indent=4)



