import os
import fitz
import matplotlib.pyplot as plt

from extractor import utils

ds_dir = r'C:\Users\fsalm\Desktop\DISTILL\LOGIC_firstbatch_163_YOLO\train'
pdf_dir = f'..{os.sep}..{os.sep}Distill.data.v2{os.sep}LOGIC'
# pdf_dir = r'C:\Users\fsalm\Desktop\DISTILL\Distill.data.v2\LOGIC'
im_lst = os.listdir(os.path.join(ds_dir, 'images'))

for im_name in im_lst:
    fname = im_name[:-4]
    fname_splits = fname.split('_')
    pdf_file = fname_splits[:-2]
    folder_name = '_'.join(pdf_file)
    sub_folder_name = fname_splits[-1]

    doc = fitz.Document((os.path.join(pdf_dir, folder_name, sub_folder_name+'.pdf')))
    p = doc[0]
    pixmap = p.get_pixmap(dpi=150)
    im = utils.pixmap_to_image(pixmap)

    plt.imshow(im)
    plt.show()

    print(im.size)

    exit()