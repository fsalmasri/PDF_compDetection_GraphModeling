import numpy as np

from extractor import doc


doc.load_pdf(pdfpath = '../Distill.data.v2/Master_Knippegroen P&ID (1).pdf')
doc.print_pdfMData()
# doc.separate_pages()
pw, ph = doc.get_pages_details()

import matplotlib.pyplot as plt
im = doc.save_image(0)
im.save(f"page_{0}.png", jpg_quality=100)
print(im)

exit()

print(np.unique(pw), np.unique(ph))