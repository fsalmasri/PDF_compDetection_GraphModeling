
import json
import os

replacement_dict = {'test_6': 'KKS-vlag', 'test_7': 'KKS-vlag', 'test_8': 'KKS-vlag', 'test_9': 'KKS-vlag'}

flst = os.listdir('LS/annots')
for f in flst:
    with open(f'LS/annots/{f}', 'r') as jf:
        annot = json.load(jf)

    for res in annot['annotations'][0]['result']:
        label = res['value']['rectanglelabels'][0]

        if label in replacement_dict:
            res['value']['rectanglelabels'][0] = replacement_dict[label]

    with open(f'LS/annots/{f}', 'w') as jf:
        json.dump(annot, jf, indent=4)
