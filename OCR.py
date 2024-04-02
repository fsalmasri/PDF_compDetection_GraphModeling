import os

import matplotlib
# matplotlib.use('QtAgg')
import matplotlib.pyplot as plt

import matplotlib.patches as patches
import numpy as np
from extractor.OCR_detector import run_OCR, run_paddle_OCR
from PIL import Image
import easyocr
from PyQt5.QtCore import QLibraryInfo
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QLibraryInfo.location(
    QLibraryInfo.PluginsPath
)

f_dir = 'data/LOGIC/0_EFF00_10CRF02.CG'
flst =np.sort(os.listdir(f_dir))

reader = easyocr.Reader(['en'], gpu=False, verbose=False)

# from paddleocr import PaddleOCR, draw_ocr
# from paddle.

# ocr_version = "PP-OCR",

# paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)



for f in flst:
    print(f)
    fpath = f'{f_dir}/{f}'

    img = Image.open(f'{fpath}/img.png')
    img = np.array(img)

    dic, bbxs = run_OCR(fpath, reader)

    # dic, bbxs = run_paddle_OCR(fpath, paddle_ocr)

    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')

    for bb in bbxs:
        rect = patches.Rectangle((bb[0], bb[1]), width=bb[2]-bb[0], height=bb[3]-bb[1], linewidth=1, edgecolor='r',
                                 facecolor='none')
        ax.add_patch(rect)

    plt.show()

