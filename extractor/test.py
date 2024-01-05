from collections import Counter
import matplotlib.pyplot as plt
import bezier
import numpy as np

from . import doc

color_mapping = {'s': 'white', 'f': 'yellow', 'fs': 'blue',
                 'qu': 'purple', 're': 'red', 'c': 'orange', 'test': 'green'}

def check_dwg_items(dwg):
    for item in dwg['items']:
        if item[0] == 'c': # or item[0] == 're' or item[0] == 'qu':
            return True

    return False

def check_PointRange(p, rng=[[100,170],[460,560]]):
    return p.x > rng[0][0] and p.x < rng[0][1] and p.y > rng[1][0] and p.y < rng[1][1]

def study_pathes():



    sp = doc.get_current_page()
    drawings = sp.single_page.get_drawings() #extended=True

    plt.figure()
    plt.imshow(sp.e_canvas)

    for dwg_idx, dwg in enumerate(drawings): #[400:]
        # flag = check_dwg_items(dwg)
        if dwg['type'] == 's':
            flag = True
        else:
            flag= False

        p_flag = False
        flag = True
        if flag:
            # print(dwg_idx, dwg)

            # print(dwg['rect'])
            # plt.figure()
            # plt.imshow(sp.e_canvas)
            for idx, path in enumerate(dwg['items']):
                if 'l' in path[0]:
                    # print(path)

                    # if path[1].x > 0 and path[2].x < 170 and path[1].y > 460 and path[2].y < 470:
                    if check_PointRange(path[1]) or check_PointRange(path[2]):
                        p_flag = True
                        plt.plot([path[1].x, path[2].x], [path[1].y, path[2].y], c=color_mapping[dwg['type']])

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

                if 'qu' in path[0]:
                    quads = path[1]

                    if (check_PointRange(quads.ul) or check_PointRange(quads.ur)
                            or check_PointRange(quads.ll) or check_PointRange(quads.lr)):
                        p_flag = True

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

                    if (check_PointRange(rect.tl) or check_PointRange(rect.tr)
                            or check_PointRange(rect.bl) or check_PointRange(rect.br)):
                        p_flag = True

                        plt.plot([rect.tl.x, rect.tr.x], [rect.tl.y, rect.tr.y],
                                 c=color_mapping['re'])
                        plt.plot([rect.tr.x, rect.br.x], [rect.tr.y, rect.br.y],
                                 c=color_mapping['re'])
                        plt.plot([rect.br.x, rect.bl.x], [rect.br.y, rect.bl.y],
                                 c=color_mapping['re'])
                        plt.plot([rect.bl.x, rect.tl.x], [rect.bl.y, rect.tl.y],
                                 c=color_mapping['re'])

                if 'c' in path[0]:
                    # print(path)

                    x_coords = [path[1].x, path[2].x, path[3].x, path[4].x]
                    y_coords = [path[1].y, path[2].y, path[3].y, path[4].y]
                    nodes = [x_coords, y_coords]

                    curve = bezier.Curve(nodes, degree=3)
                    # Sample the curve to get points for plotting
                    num_points = 100
                    curve_points = curve.evaluate_multi(np.linspace(0, 1, num_points))

                    if (check_PointRange(path[1]) or check_PointRange(path[2])
                            or check_PointRange(path[3]) or check_PointRange(path[4])):
                        p_flag = True

                        # plt.figure(figsize=(6, 6))
                        # plt.plot(x_coords, y_coords, 'ro', label='Control Points')
                        plt.plot(curve_points[0], curve_points[1], label='Bezier Curve', c=color_mapping['c'])
                        # plt.title('Cubic Bezier Curve')
                        # plt.xlabel('X-axis')
                        # plt.ylabel('Y-axis')
                        # plt.legend()
                        # plt.grid(True)
                        # plt.show()

                        # print(curve)
                        # exit()

            if p_flag:
                print(dwg_idx, dwg)

    plt.show()
        # exit()

def save_svg(filename, svg):
    with open('{filename}.svg', 'w') as f:
        f.write(svg)

def study_paths_extended():
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




def print_test():
    print(doc.pdfpath, doc.pages_count)