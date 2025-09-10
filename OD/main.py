import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO
from ultralytics import settings

import os
import torch
import fitz
from PIL import Image


ds_dir = 'LOGIC_157_YOLO_v2'

def run():
    settings.update({'runs_dir': '/runs',

                     'tensorboard': True,
                     'visualize': True,
                     'show_labels': False,
                     'datasets_dir': rf'C:\Users\fsalm\Desktop\DISTILL\{ds_dir}'})

    # Load the model.
    model = YOLO('yolov8n.pt', task='detect')
    ARGS = {
        'data': r'C:\Users\fsalm\Desktop\DISTILL\Distill\OD\ds.yaml',
        'imgsz': 1280,
        'epochs': 100,
        'batch': 8,
        'name': 'yolov8n_custom',

        'scale': 0.5,
        'mosaic': 0,
    }
    # model = YOLO('runs/detect/yolov8n_custom2/weights/best.pt', task='detect')

    # Training.
    model.train(**ARGS)


def test(no):
    model = YOLO(f'runs/detect/yolov8n_custom{no}/weights/best.pt', task='detect')
    test_dir = rf'C:\Users\fsalm\Desktop\DISTILL\{ds_dir}\test\images'

    for test_im in os.listdir(test_dir):
        results = model.predict(os.path.join(test_dir, test_im), save=False, line_width=1, conf=.4)
        result = results[0]
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        names = result.names


        print(names)

        predicted_LC_bbxs = boxes[classes == 0]
        predicted_LC_conf = confidences[classes == 0]

        predicted_LCCON_bbxs = boxes[classes == 1]
        predicted_LCCON_conf = confidences[classes == 1]

        predicted_LCInp_bbxs = boxes[classes == 2]
        predicted_LCInp_conf = confidences[classes == 2]

        predicted_txt_bbxs = boxes[classes == 3]
        predicted_txt_conf = confidences[classes == 3]

        print(len(classes), len(predicted_LC_bbxs), len(predicted_LCCON_bbxs))

        np.save('LC_bbx.npy', predicted_LC_bbxs)
        np.save('LCCON_bbx.npy', predicted_LCCON_bbxs)
        np.save('LCInp_bbx.npy', predicted_LCInp_bbxs)
        np.save('LCTXT_bbx.npy', predicted_txt_bbxs)
        exit()
        #     result.save(show_labels=False)



if __name__ == '__main__':
    # run()
    test(7)