import os
import pathlib
import numpy as np
from tqdm import tqdm
import json
from PIL import Image

from ls_ops import get_project_by_name
from ls_ops import get_tasks, create_task, get_task_by_id, get_task_by_imName,generate_ls_uid, create_annot, update_annot
from ls_ops import fill_in_annotations, fill_tasks

from LS_utils import get_labels_from_file, get_classes
from LS_connector import ds_dir


img_dir = f'{ds_dir}/images'
lbl_dir = f'{ds_dir}/labels'
classes_file = f'{ds_dir}/classes.txt'
img_ext = 'png'

imgs_lst = os.listdir(img_dir)
lbl_files = os.listdir(lbl_dir)
classes = get_classes(classes_file)


# JSON_file = 'package1_Inserted.json'
# with open('package1_Inserted.json', 'r') as fp:
#     tasks_dic = json.load(fp)
#
# print(tasks_dic)

new_proj, new_proj_id = get_project_by_name('LOGICS')

def fill_annots():
    tasks = get_tasks(new_proj_id)
    tasks_dic = {task.data['text']: task.id for task in tasks}

    for lbl_name in tqdm(lbl_files):
        root, _ = os.path.splitext(lbl_name)
        im = Image.open(os.path.join(img_dir, f'{root}.{img_ext}'))
        width, height = im.size

        task_id = tasks_dic[root]

        lbls = get_labels_from_file(lbl_dir, lbl_name)

        if len(lbls) > 0:
            results = fill_in_annotations(lbls, classes, width, height)
        else:
            results = []

        create_annot(task_id=task_id, results=results)

# fill_tasks(folder_name='LOGICS', proj_id=new_proj_id, imgs_lst=imgs_lst)
fill_annots()


