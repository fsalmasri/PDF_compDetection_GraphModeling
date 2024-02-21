
import json
import os

from labels import replacement_dict, delete_set


def clean_group_ls():
    flst = os.listdir('LS/annots')
    labels_lst = set()
    for f in flst:
        with open(f'LS/annots/{f}', 'r') as jf:
            annot = json.load(jf)

        to_delete = []
        results_lst = annot['annotations'][0]['result']

        for idx, res in enumerate(results_lst):

            label = res['value']['rectanglelabels'][0]

            if label in replacement_dict:
                res['value']['rectanglelabels'][0] = replacement_dict[label]

            elif label in delete_set:
                to_delete.append(idx)

            else:
                labels_lst.add(label)

        results_lst = [x for idx, x in enumerate(results_lst) if idx not in to_delete]

        annot['annotations'][0]['result'] = results_lst

        with open(f'LS/annots/{f}', 'w') as jf:
            json.dump(annot, jf, indent=4)


    for label in labels_lst:
        print(label)

clean_group_ls()