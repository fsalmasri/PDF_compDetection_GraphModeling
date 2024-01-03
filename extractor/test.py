from collections import Counter
import matplotlib.pyplot as plt

from . import doc



def check_dwg_items(dwg):
    for item in dwg['items']:
        if not 'l' in item :
            return True

    return False


def study_pathes():
    color_mapping = {'s': 'white', 'f': 'yellow', 'fs': 'blue'}


    sp = doc.get_current_page()
    drawings = sp.single_page.get_drawings() #extended=True

    plt.figure()
    plt.imshow(sp.e_canvas)
    for dwg in drawings:
        flag = check_dwg_items(dwg)
        if flag:
            print(dwg)
        # print(flag)
        # exit()
        # for idx, path in enumerate(dwg['items']):
        #     if 'l' in path:
        #         plt.plot([path[1].x, path[2].x], [path[1].y, path[2].y], c=color_mapping[dwg['type']])
        #     else:
        #         print(dwg)
        # plt.show()

def print_test():
    print(doc.pdfpath, doc.pages_count)