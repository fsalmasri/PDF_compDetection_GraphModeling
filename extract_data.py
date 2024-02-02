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

# flst = np.sort(os.listdir('../Distill.data.v2/PID'))
# page_number = 1
#
# for p in flst[:]:
#     page_number= int(p[:-4])
#     print(f'parsing page {page_number}')
#
#     doc.pages = []
#     doc.load_pdf(pdfpath= f'../Distill.data.v2/PID/{page_number}.pdf')
#
#     sp = doc.get_current_page()
#     sp.extract_page_info()
#     if sp.pw == 2837.0 and sp.ph == 1965.0:
#
#         sp.extract_text()
#         sp.extract_paths()
#         remove_borders()
#
#         # sp.save_images()
#         find_boundingBoxes(margin_percentage=0.25)
#         # clean_borders_svg(page_number)
#
#         print('group', len(sp.grouped_prims))
#         plotter.plot_full_dwg()
# #         # sp.load_data()
# #         sp.save_data()


flst = np.sort(os.listdir('data'))
group_feX = []
for p in flst[:]:
    page_number = int(p)

    doc.pages = []
    doc.load_pdf(pdfpath=f'../Distill.data.v2/PID/{page_number}.pdf')

    sp = doc.get_current_page()
    sp.load_data()
    feX = extrct_features()
    group_feX.extend(feX)


hist = [x[2] for x in group_feX]
hist = np.vstack(hist)
np.save('hist.npy', hist)
labels = group_clustering(hist)

print(f'len of sent histograms: {len(hist)}  unique labels: {np.unique(labels)}')

# update grouped_prims with class label
[x.append(idx) for idx, x in enumerate(group_feX)]
for p in flst:
    sub_group = [x for x in group_feX if x[0] == p]

    with open(f'data/{p}/grouped_prims.json') as jf:
        grouped_prims = json.load(jf, object_hook=keystoint)

    for g in sub_group:
        page_number, GID, _, idx = g

        grouped_prims[GID]['class'] = int(labels[idx])

    with open(f"data/{p}/grouped_prims.json", "w") as jf:
        json.dump(grouped_prims, jf, indent=4)









# exit()


# plotter.plot_full_dwg()
# sp.save_data(str(page_number))

# Detect_unconnected_letters()

# plt.show()

# exit()
# clean_duplicates_paths()
# exit()
# extractor.plotter.plot_graph_nx()
# study_line_fill_connection()
# study_disconnected_comp()
# study_buffering_by_paths()
# study_buffering_by_nodes()

# Clean_filling_strikes()
# clean_filled_strokes()

exit()


# sp.clean_data()

# sp.plot_txtblocks_regions()
exit()
# sp.plot_two_full_figures()
# sp.plot_connected_components()
#
# x = [670, 730] #[7, 9]
# y = [375, 440] #[63, 66]
# sp.find_connectedComp_inRegion(x, y)

# sp.study_connected_components()

