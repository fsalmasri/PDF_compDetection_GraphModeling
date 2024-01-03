from collections import Counter
import matplotlib.pyplot as plt
import bezier

from . import doc



def check_dwg_items(dwg):
    for item in dwg['items']:
        if item[0] == 'c':
            return True

    return False


def study_pathes():
    color_mapping = {'s': 'white', 'f': 'yellow', 'fs': 'blue', 'qu': 'purple', 're': 'red', 'cu': 'orange'}


    sp = doc.get_current_page()
    drawings = sp.single_page.get_drawings() #extended=True

    # plt.figure()
    # plt.imshow(sp.e_canvas)

    for dwg in drawings:
        flag = check_dwg_items(dwg)
        # flag = True
        if flag:
            print(dwg)
            # plt.figure()
            # plt.imshow(sp.e_canvas)
            for idx, path in enumerate(dwg['items']):
                if 'l' in path[0]:
                    plt.plot([path[1].x, path[2].x], [path[1].y, path[2].y], c=color_mapping[dwg['type']])
                # print(path)
                if 'qu' in path[0]:
                    quads = path[1]
                    plt.plot([quads.ul.x, quads.ur.x], [quads.ul.y, quads.ur.y],
                             c=color_mapping['qu'])
                    plt.plot([quads.ur.x, quads.lr.x], [quads.ur.y, quads.lr.y],
                             c=color_mapping['qu'])
                    plt.plot([quads.lr.x, quads.ll.x], [quads.lr.y, quads.ll.y],
                             c=color_mapping['qu'])
                    plt.plot([quads.ll.x, quads.ul.x], [quads.ll.y, quads.ul.y],
                             c=color_mapping['qu'])

                if 're' in path[0]:
                    rect = path[1]
                    plt.plot([rect.tl.x, rect.tr.x], [rect.tl.y, rect.tr.y],
                             c=color_mapping['re'])
                    plt.plot([rect.tr.x, rect.br.x], [rect.tr.y, rect.br.y],
                             c=color_mapping['re'])
                    plt.plot([rect.br.x, rect.bl.x], [rect.br.y, rect.bl.y],
                             c=color_mapping['re'])
                    plt.plot([rect.bl.x, rect.tl.x], [rect.bl.y, rect.tl.y],
                             c=color_mapping['re'])

                if 'c' in path[0]:
                    print(path)

                    x_coords = [path[1].x, path[2].x, path[3].x, path[4].x]
                    y_coords = [path[1].y, path[2].y, path[3].y, path[4].y]
                    nodes = [x_coords, y_coords]

                    curve = bezier.Curve(nodes, degree=3)

                    import numpy as np
                    # Sample the curve to get points for plotting
                    num_points = 100
                    curve_points = curve.evaluate_multi(np.linspace(0, 1, num_points))


                    # Extract x and y coordinates of the curve points
                    curve_x_coords = curve_points[0]
                    curve_y_coords = curve_points[1]

                    plt.figure(figsize=(6, 6))
                    plt.plot(x_coords, y_coords, 'ro', label='Control Points')
                    plt.plot(curve_x_coords, curve_y_coords, label='Bezier Curve')
                    plt.title('Cubic Bezier Curve')
                    plt.xlabel('X-axis')
                    plt.ylabel('Y-axis')
                    plt.legend()
                    plt.grid(True)
                    plt.show()

                    # print(curve)
                    # exit()

    # plt.show()


def print_test():
    print(doc.pdfpath, doc.pages_count)