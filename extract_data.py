from extractor import doc, study_line_fill_connection, study_disconnected_comp
from extractor import study_buffering_by_paths, study_buffering_by_nodes
from extractor import plot_full_dwg, Clean_filling_strikes, Detect_unconnected_letters, clean_filled_strokes
from extractor import clean_duplicates_paths
import extractor


doc.load_pdf(pdfpath = '../data/pid/61.pdf')

# restart_filled_stroke_extraction()
# exit()
# study_pathes()
# study_paths_extended()
# exit()
# doc.print_pdfMData()

sp = doc.get_current_page()
# sp.extract_text()
# sp.extract_paths()
# sp.save_data('61')

sp.load_data('61')
# clean_duplicates_paths()
# exit()
# extractor.plotter.plot_graph_nx()
# study_line_fill_connection()
# study_disconnected_comp()
# study_buffering_by_paths()
# study_buffering_by_nodes()

# plot_full_dwg(region= True)



# Clean_filling_strikes()
# clean_filled_strokes()
Detect_unconnected_letters()
# sp.build_connected_components()
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

