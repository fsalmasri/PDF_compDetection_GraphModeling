import numpy as np
import time


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

from extractor import (
    detect_LC_rectangles,
   clean_text_by_OCR_bbxs,
    detect_LC_connectors,
   detect_connections,
   # Convert_to_LS_data,
   # assign_tags,

   # convert_tags_to_graphs
                        )

# from PyQt5.QtCore import QLibraryInfo
# os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QLibraryInfo.location(
#     QLibraryInfo.PluginsPath
# )
#     if sp.pw == 2837.0 and sp.ph == 1965.0:


def extract_data():
    folders_lst = np.sort(os.listdir(extractor.Data_load_path))
    AOI = [60, 60, 780, 500]  # x0,y0,x1,y1

    for folder in folders_lst[:1]:

        print(folder)
        current_path = f'{extractor.Data_load_path}/{folder}'
        flst = np.sort(os.listdir(current_path))

        print(f'Loading folder: {folder} | Number of files: {len(flst)}')
        for p in flst[0:]:
            page_number = int(p[:-4])

            doc.pages = []
            doc.load_pdf(pdfpath= f'{current_path}/{p}')

            sp = doc.get_current_page()
            sp.extract_page_info()
            print(f'parsing page {page_number} info {sp.page_info}')

            sp.save_images(dpi=300)

            start_time = time.time()
            sp.extract_text()
            print(f"Text Extraction from PDF blocks --- minutes --- {(time.time() - start_time) / 60}")

            start_time = time.time()
            save_OCR_bbx(fpath=sp.pdf_saving_path, cuda=True)
            print(f"OCR --- minutes --- {(time.time() - start_time) / 60}")

            start_time = time.time()
            # TODO find AOI
            #TODO check what was update_primitives
            sp.extract_paths(OCR_cleaner=False, AOI=AOI)
            print(f"Paths --- minutes --- {(time.time() - start_time) / 60}")

            start_time = time.time()
            sp.update_primitives_tables()
            print(f"primitives --- minutes --- {(time.time() - start_time)/60}")

            sp.save_data()


            # PID old functions
            # remove_borders()
            # find_boundingBoxes(margin_percentage=0.25)
            # clean_borders_svg(page_number)
            # sp.load_data()
            # plotter.plot_txtblocks_regions()
            # plotter.plot_full_dwg(paths=True, connected_com=False)


            # exit()

def process_data():
    # TODO Move this somewhere else
    from pathlib import Path
    # Path(extractor.folder_to_save_annots).mkdir(parents=True, exist_ok=True)
    # Path(extractor.folder_to_save_images).mkdir(parents=True, exist_ok=True)

    folders_lst = np.sort(os.listdir(extractor.Saving_path))
    print(len(folders_lst))
    for folder in folders_lst[:]:
        print(f'Processing folder: {folder}')

        current_path = f'{extractor.Data_load_path}/{folder}'
        flst = np.sort(os.listdir(current_path))

        for p in flst[1:]:
            page_number= int(p[:-4])
            print(f'parsing page {page_number}')

            doc.pages = []
            doc.load_pdf(pdfpath= f'{current_path}/{p}')

            sp = doc.get_current_page()
            sp.extract_page_info()
            sp.load_data()


            # plot full drawing in groups.
            # plotter.plot_full_dwg(paths=True, connected_com=False, OCR_boxs=True)

            # clean_text_by_OCR_bbxs(save_LUTs=True, plot=False)
            # detect_LC_rectangles(save_LUTs=True, plot=False)
            # detect_LC_connectors(save_LUTs=True, plot=False)
            detect_connections(save_LUTs=False, plot=True)
            # tags_dictionary = assign_tags(plot=False)
            # convert_tags_to_graphs(tags_dictionary)
            # Convert_to_LS_data(include_text=True)


            # plotter.plot_grouped_primes(LC=True, LC_input=True, LC_con=True, Con=False, bbx=False)

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







