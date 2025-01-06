import os
import random
import shutil
from pathlib import Path


ds_dir = r'../../../LOGICS/LOGIC 295_v2/project-11-at-2024-11-25-12-02-27e3dba2'
imgs_lst = os.listdir(os.path.join(ds_dir, 'images'))


random.shuffle(imgs_lst)
train_lst = imgs_lst[:int(len(imgs_lst)*.7)]
val_lst = imgs_lst[int(len(imgs_lst)*.7):int(len(imgs_lst)*.7)+int(len(imgs_lst)*.2)]
test_lst = imgs_lst[int(len(imgs_lst)*.7)+int(len(imgs_lst)*.2):]

print(len(imgs_lst), len(train_lst), len(val_lst), len(test_lst))

Path(os.path.join(ds_dir, 'train', 'images')).mkdir(parents=True, exist_ok=True)
Path(os.path.join(ds_dir, 'train', 'labels')).mkdir(parents=True, exist_ok=True)
for im_name in train_lst:
    fname = im_name[:-3]
    lbl_name = fname + 'txt'
    shutil.copy(os.path.join(ds_dir, 'images', im_name), os.path.join(ds_dir, 'train', 'images', im_name))
    shutil.copy(os.path.join(ds_dir, 'labels', lbl_name), os.path.join(ds_dir, 'train', 'labels', lbl_name))


Path(os.path.join(ds_dir, 'valid', 'images')).mkdir(parents=True, exist_ok=True)
Path(os.path.join(ds_dir, 'valid', 'labels')).mkdir(parents=True, exist_ok=True)
for im_name in val_lst:
    fname = im_name[:-3]
    lbl_name = fname + 'txt'
    shutil.copy(os.path.join(ds_dir, 'images', im_name), os.path.join(ds_dir, 'valid', 'images', im_name))
    shutil.copy(os.path.join(ds_dir, 'labels', lbl_name), os.path.join(ds_dir, 'valid', 'labels', lbl_name))


Path(os.path.join(ds_dir, 'test', 'images')).mkdir(parents=True, exist_ok=True)
Path(os.path.join(ds_dir, 'test', 'labels')).mkdir(parents=True, exist_ok=True)
for im_name in test_lst:
    fname = im_name[:-3]
    lbl_name = fname + 'txt'
    shutil.copy(os.path.join(ds_dir, 'images', im_name), os.path.join(ds_dir, 'test', 'images', im_name))
    shutil.copy(os.path.join(ds_dir, 'labels', lbl_name), os.path.join(ds_dir, 'test', 'labels', lbl_name))