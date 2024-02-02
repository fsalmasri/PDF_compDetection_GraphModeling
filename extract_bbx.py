
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm


flst = sorted(os.listdir('LS/annots'))
for f in tqdm(flst):
    with open(f'LS/annots/{f}', 'r') as jf:
        annot = json.load(jf)

    img = np.array(Image.open(f'data/{f[:-5]}/{f[:-4]}png'))
    image_width, image_height = img.shape[1], img.shape[0]

    for idx, res in enumerate(annot['annotations'][0]['result']):
        x = int((res['value']['x']/100)*image_width)
        y = int((res['value']['y']/100)*image_height)
        w = int((res['value']['width']/100)*image_width)
        h = int((res['value']['height']/100)*image_height)
        label = res['value']['rectanglelabels'][0]
        cropped = img[y:y+h, x:x+w, :]


        fname = f'{f[:-5]}_{idx}.png'
        path_to_save = f'../verify/{label}'
        Path(f"{path_to_save}").mkdir(parents=True, exist_ok=True)
        Image.fromarray(cropped).save(f'{path_to_save}/{fname}')
