import os

import numpy as np
from PIL import Image
import uuid
import secrets
import string
from tqdm import tqdm

from LS_utils import get_labels_from_file, get_classes
from LS_connector import ls



# imgs_dir = r'C:\Users\fsalm\Desktop\projects\ULB\UGCT\UG_CT_images\imgs'
# labels_dir = r'C:\Users\fsalm\Desktop\projects\ULB\UGCT\UG_CT_images\labels'
# classes_file = r'C:\Users\fsalm\Desktop\projects\ULB\UGCT\UG_CT_images\classes.txt'

def create_project():
    pass

def get_project_by_name(project_name):


    projects = ls.projects.list()
    project_list = list(projects)
    number_of_projects = len(project_list)

    print(f'Number of projects in the server: {number_of_projects}')
    for proj in project_list:
        if proj.title == project_name:
            return ls.projects.get(id=proj.id), proj.id


# def get_task_by_imName(proj_id, task_im_name):
#     tasks = ls.tasks.list(project=proj_id)
#     tasks_list = list(tasks)
#
#     for task in tasks_list:
#         if task.data['text'] == task_im_name:
#             return ls.tasks.get(id=task.id)

def get_task_by_imName(tasks, task_im_name):
    tasks_list = list(tasks)

    for task in tasks_list:
        if task.data['text'] == task_im_name:
            # print('founded')
            return ls.tasks.get(id=task.id)

    return None


def get_tasks(proj_id):
    tasks = ls.tasks.list(project=proj_id)

    # for idx, v_task in enumerate(tasks):
    #     print(idx, v_task.id)
    # exit()
    # import copy
    #
    # tasks_list = list(copy.deepcopy(tasks))
    # number_of_tasks = len(tasks_list)
    #
    # print(f'Number of tasks in project {proj_id}: {number_of_tasks}')

    return tasks

def get_task_by_id(task_id):
    return ls.tasks.get(id=task_id)



def get_annot_by_id(annot_id):
    return ls.annotations.get(id=annot_id)

def create_task(proj_id, folder_name, img_name, text='', internal=False):
    if internal:
        return ls.tasks.create(
            project=proj_id,
            data={"image": f"/data/upload/{folder_name}/{img_name}", "text": f"{text}"}
        )
    else:
        return ls.tasks.create(
            project=proj_id,
            data={"image": f"data/local-files?d={folder_name}/{img_name}", "text": f"{text}"}
        )


def update_task(proj_id, task_id, pkg_name, img_name):
    ls.tasks.update(
            id=task_id,
            data={"image": f"data/local-files?d={pkg_name}/{img_name}", "text": f"{img_name}"},
            project=proj_id,
        )


def create_annot(task_id, results):
    ls.annotations.create(
        id=task_id,
        result=results,
        was_cancelled=False,
        ground_truth=True,
    )

def update_annot(task_id, annot_id, results):

    ls.annotations.update(
        id=annot_id,
        result=results,
        task=task_id,
        was_cancelled=False,
        ground_truth=True
    )


def add_tasks_from_folder(proj_id, imgs_dir, folder_name):
    flst = os.listdir(imgs_dir)

    for f in flst:
        create_task(proj_id, folder_name, img_name=f)

def generate_ls_uid(length=10):
    first_char = secrets.choice(string.ascii_lowercase)
    other_chars = ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(10))

    return f'{first_char}_{other_chars}'

def fill_tasks(folder_name, proj_id, imgs_lst):
    for im_name in imgs_lst:
        im_txt = os.path.splitext(im_name)[0]
        ls_task = create_task(proj_id, folder_name, im_name, text=im_txt, internal=False)

def fill_in_annotations(data, classes, im_w, im_h):
    results = []
    for item in data:
        class_name = classes[int(item[0])]

        width = item[3]
        height = item[4]
        x = item[1] - (width / 2)
        y = item[2] - (height / 2)

        x *= 100
        y *= 100
        width *= 100
        height *= 100

        results.append({
                    "id": f"{generate_ls_uid()}",
                    "type": "rectanglelabels",

                    "value": {
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                    "rotation": 0,
                    "rectanglelabels": [class_name]
                    },
                    "origin": "manual",
                    "to_name": "image",
                    "from_name": "label",

                    "image_rotation": 0,
                    "original_width": im_w,
                    "original_height": im_h
                })

    return results

def add_annotations_from_tasks(proj_id):
    classes = get_classes(classes_file)

    tasks = get_tasks(proj_id)
    for task in tqdm(tasks):
        task_id = task.id
        im_name = task.data['text']

        root, _ = os.path.splitext(im_name)
        lbl_name = f'{root}.txt'

        img = np.array(Image.open(os.path.join(imgs_dir, im_name)))
        im_h, im_w, _ = img.shape

        data = get_labels_from_file(labels_dir, lbl_name)
        if data:
            results = fill_in_annotations(data, classes, im_w, im_h)
        else:
            results = []
        ls.annotations.create(
            id=task_id,
            result= results,
            was_cancelled=False,
            ground_truth=True,
        )



def verify_noexist_label_annots():
    wrong_labels = ['test', 'Surge Arrestor', 'Lightning Arrestor', 'Indicating Light', 'Plug or Socket', 'Luminaire', 'Flexible Connection', 'Meter', 'Generator', 'Surge Suppresor']
    tasks = get_tasks(proj_id)
    for task in tqdm(tasks):
        task_id = task.id
        # print(f'running task: {task_id}')

        # with open('test.txt', "a+") as file:
        #     file.write(f'running task: {task_id}\n')

        annotations = task.annotations[0]
        for annot in annotations['result']:
            annot_label = annot['value']['rectanglelabels'][0]

            # with open('test.txt', "a+") as file:
            #     file.write(f"{annot_label}\n")

            annot_id = annot['id']
            if annot_label in wrong_labels:
                print(annot_label, annot_id, task_id)
        # exit()


