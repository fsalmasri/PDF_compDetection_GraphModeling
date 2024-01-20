from extractor import doc, study_line_fill_connection, study_disconnected_comp
from extractor import study_buffering_by_paths, study_buffering_by_nodes
from extractor import Clean_filling_strikes, Detect_unconnected_letters, clean_filled_strokes
from extractor import clean_duplicates_paths
from extractor import remove_borders, find_boundingBoxes

import extractor
from extractor import plotter



import matplotlib.pyplot as plt
page_number = 0

doc.load_pdf(pdfpath= f'../Distill.data.v2/PID/{page_number}.pdf')

sp = doc.get_current_page()
# sp.extract_text()
# sp.extract_paths()
# Detect_unconnected_letters()
# remove_borders()
# sp.save_data(str(page_number))


sp.load_data(str(page_number))
plotter.plot_full_dwg()
find_boundingBoxes()

plt.show()

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

