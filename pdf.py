import fitz
from page import page


class pdf():
    def __init__(self, pdfpath):

        print(fitz.__doc__)

        self.pdfpath = pdfpath
        self.pages = []
        self.pages_count = 0
        self.current_page = 0 #61

        self.load_pdf()
        self.extract_pages(save=False)


    def load_pdf(self):
        self.doc = fitz.open(self.pdfpath)

    def extract_singlePage(self, pn):
        return self.doc.load_page(pn)

    def extract_pages(self, save=False):

        for i, p in enumerate(self.doc):
            if save:
                with fitz.open() as doc_tmp:
                    doc_tmp.insert_pdf(self.doc, from_page=i, to_page=i, rotate=-1, show_progress=False)
                    doc_tmp.save(f'../data/{i}.pdf')

            self.pages.append(page(p))
        self.pages_count = len(self.pages)

        print(f'{self.pages_count} pages found')


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

