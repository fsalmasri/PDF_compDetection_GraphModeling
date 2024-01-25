import fitz
from . import Data_load_path

from .page import page

class pdf():
    def __init__(self):

        print(fitz.__doc__)

        self.pdfpath = None
        self.pages = []
        self.pages_count = 0
        self.current_page = 0


    def load_pdf(self, pdfpath):
        import os

        self.pdfpath = pdfpath
        self.doc = fitz.open(self.pdfpath)
        self.extract_pages(os.path.basename(pdfpath)[:-4])


    def extract_singlePage(self, pn):
        return self.doc.load_page(pn)

    def extract_pages(self, i):
        # for i, p in enumerate(self.doc):
        #     self.pages.append(page(p, i))
        for p in self.doc:
            self.pages.append(page(p, i))
        self.pages_count = len(self.pages)

        print(f'{self.pages_count} pages found')

    def separate_pages(self):
        for i, p in enumerate(self.doc):
            with fitz.open() as doc_tmp:
                doc_tmp.insert_pdf(self.doc, from_page=i, to_page=i, rotate=-1, show_progress=False)
                doc_tmp.save(f'{Data_load_path}/PID/{i}.pdf')

    def print_pdfMData(self):

        doc_meta = self.doc.metadata
        for k, v in doc_meta.items():
            print(k, doc_meta[k])
        print(f'page counts : {self.pages_count}')
        # print(self.doc.metadata)

    def get_pages_details(self):
        pw, ph = [], []
        for page in self.pages:
            pw.append(page.pw)
            ph.append(page.ph)

        return pw, ph


    def get_current_page(self, page_number=None):
        if not page_number:
            return self.pages[self.current_page]
        else:
            return self.pages[page_number]


    def save_image(self, pid):
        sp = self.get_current_page(pid)
        return sp.single_page.get_pixmap()


