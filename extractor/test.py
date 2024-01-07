from collections import Counter
import matplotlib.pyplot as plt
import bezier
import numpy as np

from . import doc



def check_dwg_items(dwg):
    for item in dwg['items']:
        if item[0] == 'c': # or item[0] == 're' or item[0] == 'qu':
            return True

    return False

def check_PointRange(p, rng=[[100,170],[460,560]]):
    return p.x > rng[0][0] and p.x < rng[0][1] and p.y > rng[1][0] and p.y < rng[1][1]


def plot_lines(paths_lst, dwg_type):
    for path in paths_lst:
        # print('here')
        # print(path)
        plt.plot([path[0][0], path[1][0]], [path[0][1], path[1][1]], c=color_mapping[dwg_type])




# def draw_rect():
    # if dwg['type'] == 'f':
    #     rect = dwg['rect']
    #     plt.plot([rect.tl.x, rect.tr.x], [rect.tl.y, rect.tr.y],
    #              c=color_mapping['test'])
    #     plt.plot([rect.tr.x, rect.br.x], [rect.tr.y, rect.br.y],
    #              c=color_mapping['test'])
    #     plt.plot([rect.br.x, rect.bl.x], [rect.br.y, rect.bl.y],
    #              c=color_mapping['test'])
    #     plt.plot([rect.bl.x, rect.tl.x], [rect.bl.y, rect.tl.y],
    #              c=color_mapping['test'])

def study_pathes():

    sp = doc.get_current_page()
    drawings = sp.single_page.get_drawings()

    plt.figure()
    plt.imshow(sp.e_canvas)

    for dwg_idx, dwg in enumerate(drawings):
        dwg_items = dwg['items']
        dwg_type = dwg['type']
        dwg_rect = dwg['rect']

        item_paths = []
        def add_to_main(p1, p2, item_t):
            item_paths.append({'p1': [p1[0], p1[1]], 'p2': [p2[0], p2[1]],
                               'item_type': item_t, 'path_type': dwg_type})

        for idx, item in enumerate(dwg_items):
            if item[0] == 'l':
                add_to_main([item[1].x, item[1].y], [item[2].x, item[2].y], item[0])

            if item[0] == 'qu':
                quads = item[1]

                add_to_main([quads.ul.x, quads.ul.y], [quads.ur.x, quads.ur.y], item[0])
                add_to_main([quads.ur.x, quads.ur.y], [quads.lr.x, quads.lr.y], item[0])
                add_to_main([quads.lr.x, quads.lr.y], [quads.ll.x, quads.ll.y], item[0])
                add_to_main([quads.ll.x, quads.ll.y], [quads.ul.x, quads.ul.y], item[0])

            if item[0] == 're':
                rect = item[1]
                add_to_main([rect.tl.x, rect.tl.y], [rect.tr.x, rect.tr.y], item[0])
                add_to_main([rect.tr.x, rect.tr.y], [rect.br.x, rect.br.y], item[0])
                add_to_main([rect.br.x, rect.br.y], [rect.bl.x, rect.bl.y], item[0])
                add_to_main([rect.bl.x, rect.bl.y], [rect.tl.x, rect.tl.y], item[0])

            if item[0] == 'c':
                x_coords = [item[1].x, item[2].x, item[3].x, item[4].x]
                y_coords = [item[1].y, item[2].y, item[3].y, item[4].y]
                nodes = [x_coords, y_coords]

                curve = bezier.Curve(nodes, degree=3)
                num_points = 10
                curve_points = curve.evaluate_multi(np.linspace(0, 1, num_points))
                curve_points = np.array(curve_points).T
                for i in range(curve_points.shape[0] - 1): add_to_main(curve_points[i], curve_points[i + 1], item[0])


        if len(item_paths) > 0 and dwg_type == 'f' and item_paths[-1]['item_type'] == 'l':
            item_paths.append({'p1':[item_paths[-1]['p2'][0], item_paths[-1]['p2'][1]],
                               'p2':[item_paths[0]['p1'][0], item_paths[0]['p1'][1]],
                               'item_type': 'l', 'path_type': dwg_type})

        plot_items(item_paths)

        # plot_lines(paths_lst, dwg_type)
        # plot_lines(qu_lst, 'qu')
        # plot_lines(rect_lst, 're')
        # plot_lines(cu_lst, 'c')

    plt.show()

def save_svg(filename, svg):
    with open('{filename}.svg', 'w') as f:
        f.write(svg)

def study_paths_svg():
    sp = doc.get_current_page()

    # from PIL import Image
    # import cairosvg
    # from io import BytesIO
    # import pysvg
    # import xml.etree.ElementTree as ET
    from xml.dom import minidom
    from svg.path import parse_path
    from svg.path.path import Line, Move, CubicBezier

    svg_image = sp.single_page.get_svg_image()
    svg_dom = minidom.parseString(svg_image)


    clip_paths = svg_dom.getElementsByTagName('clipPath')
    g_tag = svg_dom.getElementsByTagName('g')

    for g in g_tag:
        group_id = g.getAttribute('id')
        path_elements = g.getElementsByTagName('path')

        plt.figure()
        plt.imshow(sp.e_canvas)

        # Iterate through path elements
        for path_element in path_elements[:]:
            path_data = path_element.getAttribute('d')  # Get the 'd' attribute containing the path data

            transform_matrix_data = path_element.getAttribute('transform')
            transform_matrix = list(map(float, transform_matrix_data[7:-1].split(',')))

            transform_matrix = np.array([[transform_matrix[0], transform_matrix[2], transform_matrix[4]],
                                         [transform_matrix[1], transform_matrix[3], transform_matrix[5]],
                                         [0, 0, 1]])

            parsed_path = parse_path(path_data)

            # Print the clipPath ID and path data line by line
            # print(f"Group ID: {group_id}")
            # print(f"Path Data: {path_data}")
            # print(f"Parsed Path Data: {parsed_path}")
            # print(transform_matrix)

            def transform_points(x, y, transform_matrix):
                point_vector = np.array([x, y, 1])
                transformed_point = np.dot(transform_matrix, point_vector)
                transformed_x, transformed_y, _ = transformed_point

                return transformed_x, transformed_y

            def check_PointRange_2(x, y, rng=[[140, 151], [490, 510]]): #[100, 170], [460, 560]
                return x > rng[0][0] and x < rng[0][1] and y > rng[1][0] and y < rng[1][1]

            flag = False
            for e in parsed_path:
                if isinstance(e, Line):
                    x0 = e.start.real
                    y0 = e.start.imag
                    x1 = e.end.real
                    y1 = e.end.imag

                    # print(path_data, transform_matrix_data)
                    # print("(%.2f, %.2f) - (%.2f, %.2f)" % (x0, y0, x1, y1))

                    x0, y0 = transform_points(x0, y0, transform_matrix)
                    x1, y1 = transform_points(x1, y1, transform_matrix)

                    # print("(%.2f, %.2f) - (%.2f, %.2f)" % (x0, y0, x1, y1))

                    if check_PointRange_2(x0, y0) and check_PointRange_2(x1, y1):
                        flag = True
                        # print(e)
                        print("(%.2f, %.2f) - (%.2f, %.2f)" % (x0, y0, x1, y1))
                        plt.plot([x0, x1], [y0, y1], c='white')


                elif isinstance(e, Move):
                    pass

                elif isinstance(e, CubicBezier):
                    pass

                else:
                    pass

            if flag:

                import svgpathtools

                print(path_element.attributes.items())
                print(svgpathtools.parse_path(path_data))
                exit()
                # for e in parsed_path:
                #     print(e)

                print("--------------------")


        plt.show()
        exit()

    exit()



