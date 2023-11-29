import utils
from pdf import pdf

pdfpath = '../data/DBYD_4NT_0813680_000_00_TOTALDOC (3).pdf'
doc = pdf(pdfpath)


# doc.print_pdfMData()

sp = doc.get_current_page()

# sp.extract_paths()
sp.load_data()

# sp.plot_two_full_figures()
# sp.plot_connected_components()
#
# x = [670, 730] #[7, 9]
# y = [375, 440] #[63, 66]
# sp.find_connectedComp_inRegion(x, y)

sp.study_connected_components()

