from extractor import doc, study_pathes



doc.load_pdf(pdfpath = '../data/pid/61.pdf')

study_pathes()
exit()
# doc.print_pdfMData()

sp = doc.get_current_page()
# sp.extract_text()
sp.extract_paths()



# sp.save_data('61')
# sp.load_data('61')
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

