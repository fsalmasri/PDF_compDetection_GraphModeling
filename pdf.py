import fitz
from page import page


class pdf():
    def __init__(self, pdfpath):

        print(fitz.__doc__)

        self.pdfpath = pdfpath
        self.pages = []
        self.pages_count = 0
        self.current_page = 61

        self.load_pdf()
        self.extract_pages()


    def load_pdf(self):
        self.doc = fitz.open(self.pdfpath)

    def extract_singlePage(self, pn):
        return self.doc.load_page(pn)

    def extract_pages(self):
        for p in self.doc:
            self.pages.append(page(p))

        self.pages_count = len(self.pages)


    def print_pdfMData(self):

        doc_meta = self.doc.metadata
        for k, v in doc_meta.items():
            print(k, doc_meta[k])
        print(f'page counts : {self.pages_count}')
        # print(self.doc.metadata)

    def get_current_page(self, page_number=None):
        if not page_number:
            return self.pages[self.current_page]
        else:
            return self.pages[page_number]

