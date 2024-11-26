import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO
from ultralytics import settings

import os
import torch
# import fitz
from PIL import Image


# ds_dir = 'LOGIC_157_YOLO_v2'

def run():
    # model_name = 'yolov8n_custom1'
    saving_model_name = 'yolov8n_custom12'

    settings.update({
        'runs_dir': '/runs',

        'tensorboard': True,
        'visualize': True,
        'show_labels': False,
        'visualize': True
    })

    # Load the model.
    # model = YOLO('yolo11n.pt', task='detect')
    model = YOLO('yolo11-obb.yaml', task='detect')

    results = model.train(
        data='/home/feras/Desktop/DISTILL/Distill/OD/ds.yaml',
        imgsz=1280,
        epochs=1000,
        batch=24,
        workers=8,
        device=[0, 1],

        pose=0,
        kobj=0,
        cls=1,
        box= 10,
        dfl= 0.5,

        lr0=0.001,

        dropout=0.4,
        scale=0.2,
        mosaic=1,
        fliplr=False,
        show_labels=False,
        show_conf = True,

        name=f'{saving_model_name}',
        # resume=True,
        save=True,

        line_width=1,

    )
    # Training.
    # model.train(**ARGS)


def test(no):
    model = YOLO(f'runs/detect/yolov8n_custom{no}/weights/best.pt', task='detect')
    test_dir = f'../../Distill_logics_cropped/test/images/'

    for test_im in os.listdir(test_dir):
        results = model.predict(os.path.join(test_dir, test_im), save=False, line_width=1, conf=.1)
        result = results[0]
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        names = result.names

        #
        print(names)
        print(classes)
        exit()

        # predicted_LC_bbxs = boxes[classes == 0]
        # predicted_LC_conf = confidences[classes == 0]
        #
        # predicted_LCCON_bbxs = boxes[classes == 1]
        # predicted_LCCON_conf = confidences[classes == 1]
        #
        # predicted_LCInp_bbxs = boxes[classes == 2]
        # predicted_LCInp_conf = confidences[classes == 2]
        #
        # predicted_txt_bbxs = boxes[classes == 3]
        # predicted_txt_conf = confidences[classes == 3]
        #
        # print(len(classes), len(predicted_LC_bbxs), len(predicted_LCCON_bbxs))
        #
        # np.save('LC_bbx.npy', predicted_LC_bbxs)
        # np.save('LCCON_bbx.npy', predicted_LCCON_bbxs)
        # np.save('LCInp_bbx.npy', predicted_LCInp_bbxs)
        # np.save('LCTXT_bbx.npy', predicted_txt_bbxs)
        # exit()
        #     result.save(show_labels=False)



if __name__ == '__main__':
    # run()
    test(126)