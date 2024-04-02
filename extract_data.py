import numpy as np

from extractor import doc, study_line_fill_connection, study_disconnected_comp
from extractor import study_buffering_by_paths, study_buffering_by_nodes
from extractor import Clean_filling_strikes, Detect_unconnected_letters, clean_filled_strokes
from extractor import clean_duplicates_paths
from extractor import remove_borders, find_boundingBoxes, study_clustering, extrct_features, group_clustering

from extractor.svg_extraction import clean_borders_svg
import extractor
from extractor import plotter
import matplotlib.pyplot as plt
import os
import json

from extractor.utils import keystoint
from extractor.OCR_detector import save_OCR_bbx

from extractor import detect_rectangles

from PyQt5.QtCore import QLibraryInfo
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QLibraryInfo.location(
    QLibraryInfo.PluginsPath
)

folders_lst = np.sort(os.listdir(extractor.Data_load_path))
AOI = [60, 60, 780, 500] # x0,y0,x1,y1


# for folder in folders_lst[3:5]:
#     print(folder)
#
#     current_path = f'{extractor.Data_load_path}/{folder}'
#     flst = np.sort(os.listdir(current_path))
#
#     for p in flst[0:]:
#         page_number= int(p[:-4])
#         print(f'parsing page {page_number}')
#
#         pdf_path = f'{current_path}/{p}'
#         doc.pages = []
#         doc.load_pdf(pdfpath= f'{current_path}/{p}')
#
#         sp = doc.get_current_page()
#         sp.extract_page_info()
#         # print(sp.page_info)
#     #     if sp.pw == 2837.0 and sp.ph == 1965.0:
#
#         # sp.save_images(pdfpath=f'{current_path}/{p}', dpi=300)
#         # sp.extract_text()
#
#         # save_OCR_bbx(fpath=sp.pdf_saving_path, cuda=False)
#         # sp.extract_paths(OCR_cleaner=False, AOI=AOI)
#         # sp.save_data()
#
#                 # remove_borders()
#     #
#
#             # find_boundingBoxes(margin_percentage=0.25)
#             # clean_borders_svg(page_number)
#     #
#         sp.load_data()
#         # plotter.plot_txtblocks_regions()
#         plotter.plot_full_dwg(paths=True, connected_com=False)
#
#         exit()
#     # exit()
# exit()


folder  = '1002_EFF01_10HHG43CF001'
current_path = f'{extractor.Data_load_path}/{folder}'
flst = np.sort(os.listdir(current_path))

for p in flst[0:]:
    page_number= int(p[:-4])
    print(f'parsing page {page_number}')

    pdf_path = f'{current_path}/{p}'
    doc.pages = []
    doc.load_pdf(pdfpath= f'{current_path}/{p}')

    sp = doc.get_current_page()
    sp.extract_page_info()
    sp.load_data()
    # plotter.plot_full_dwg(paths=False, connected_com=True, OCR_boxs=False)

    detect_rectangles()


    exit()




# group_feX = []
# for p in folders_lst[3:5]:
#     page_number = int(p)
#
#     doc.pages = []
#     doc.load_pdf(pdfpath=f'../Distill.data.v2/PID/{page_number}.pdf')
#
#     sp = doc.get_current_page()
#     sp.load_data()
#     feX = extrct_features()
#     group_feX.extend(feX)
#
#
# hist = [x[2] for x in group_feX]
# hist = np.vstack(hist)
# np.save('hist.npy', hist)
# labels = group_clustering(hist)
#
# print(f'len of sent histograms: {len(hist)}  unique labels: {np.unique(labels)}')
#
# # update grouped_prims with class label
# [x.append(idx) for idx, x in enumerate(group_feX)]
# for p in flst:
#     sub_group = [x for x in group_feX if x[0] == p]
#
#     with open(f'data/{p}/grouped_prims.json') as jf:
#         grouped_prims = json.load(jf, object_hook=keystoint)
#
#     for g in sub_group:
#         page_number, GID, _, idx = g
#
#         grouped_prims[GID]['class'] = int(labels[idx])
#
#     with open(f"data/{p}/grouped_prims.json", "w") as jf:
#         json.dump(grouped_prims, jf, indent=4)







