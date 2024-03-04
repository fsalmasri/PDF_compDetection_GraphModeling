import os
from glob import glob
import shutil
from extractor import doc
from pathlib import Path
from tqdm import tqdm

# extract all pdf files form subfolders and group them in one folder. counter addded to file names.

# logic_main_dir = '../Distill.data.v2/logic_and_HMI/Gent'
#
# flst = os.walk(logic_main_dir)
#
# files_list = []
# for root, dirs, files in flst:
#     for file in files:
#         files_list.append(os.path.join(root, file))
#
#
# for idx, f in enumerate(files_list):
#     fname = f'{idx}_{os.path.basename(f)}'
#     shutil.copy(f, f'../Distill.data.v2/Logic_grouped/{fname}')





# split pdf files into separate pages and save them in folders named with the main pdf file name

pdf_grouped_dir = '../Distill.data.v2/Logic_grouped'
pdf_dir = '../Distill.data.v2/LOGIC'
pdf_lst = os.listdir(pdf_grouped_dir)

for pdf_file in tqdm(pdf_lst[:]):
    doc.load_pdf(pdfpath=f'{pdf_grouped_dir}/{pdf_file}')
    doc.separate_pages()

