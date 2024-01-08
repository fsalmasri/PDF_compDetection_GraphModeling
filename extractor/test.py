from collections import Counter
import matplotlib.pyplot as plt
import bezier
import numpy as np
from shapely.geometry import LineString, Point


from . import doc
from . import plotter

from .utils import return_primitives_by_node, return_pathsIDX_given_nodes, return_paths_given_nodes


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


def study_buffering_by_paths():

    sp = doc.get_current_page()

    connected_primitives = [prim_v for prim_k, prim_v in sp.primitives.items() if len(prim_v) > 2]
    # unconnected_nodes = [sp.nodes_LUT[item] for prim_k, prim_v in sp.primitives.items() if len(prim_v) < 3 for item in prim_v]
    unconnected_nodes = [prim_v for prim_k, prim_v in sp.primitives.items() if len(prim_v) < 3]

    fig, ax = plt.subplots()
    plt.imshow(sp.e_canvas)

    for prim_v in connected_primitives[150:300]:
        paths = return_paths_given_nodes(prim_v, sp.paths_lst, sp.nodes_LUT,
                                 replace_nID=True)

        path_geometries = [LineString([tuple(p['p1']), tuple(p['p2'])]) for p in paths]

        buffer_distance = 0.5  # Adjust the buffer distance as needed
        path_buffers = [path.buffer(buffer_distance) for path in path_geometries]


        # fig, ax = plt.subplots()
        # plt.imshow(sp.e_canvas)
        plotter.plot_items(paths)

        for buffered_path in path_buffers:
            x, y = buffered_path.exterior.xy
            ax.plot(x, y, color='pink')  # Adjust color and styling as needed

        for nodes in unconnected_nodes:
            node = [sp.nodes_LUT[nodes[0]], sp.nodes_LUT[nodes[1]]]

            node_point_1 = Point(node[0])
            node_point_2 = Point(node[1])

            for buffer_zone in path_buffers:
                if node_point_1.within(buffer_zone) or node_point_2.within(buffer_zone):
                    # print(f"Node {node} falls within a buffer zone. of {paths}")

                    unconnected_paths = return_paths_given_nodes(nodes, sp.paths_lst, sp.nodes_LUT,
                                                                 replace_nID=True, test='test')
                    plotter.plot_items(unconnected_paths)

                    # ax.scatter(nodes[0], nodes[1], s=120, marker='*', color='gold', zorder=3)


    plt.show()


def study_buffering_by_nodes():

    sp = doc.get_current_page()

    connected_primitives = [prim_v for prim_k, prim_v in sp.primitives.items() if len(prim_v) > 2]
    unconnected_nodes = [prim_v for prim_k, prim_v in sp.primitives.items() if len(prim_v) < 3]

    fig, ax = plt.subplots()
    plt.imshow(sp.e_canvas)

    for prim_v in connected_primitives[250:4000]:

        paths = return_paths_given_nodes(prim_v, sp.paths_lst, sp.nodes_LUT, replace_nID=True)

        if paths[0]['item_type'] == 'l':

            node_buffer_distance = 0.1
            node_buffers = [Point(sp.nodes_LUT[node]).buffer(node_buffer_distance) for node in prim_v]

            # fig, ax = plt.subplots()
            # plt.imshow(sp.e_canvas)
            plotter.plot_items(paths)

            for buffered_path in node_buffers:
                x, y = buffered_path.exterior.xy
                ax.plot(x, y, color='pink')  # Adjust color and styling as needed

            for nodes in unconnected_nodes:
                node = [sp.nodes_LUT[nodes[0]], sp.nodes_LUT[nodes[1]]]

                node_point_1 = Point(node[0])
                node_point_2 = Point(node[1])

                for buffer_zone in node_buffers:
                    if node_point_1.within(buffer_zone) or node_point_2.within(buffer_zone):
                        # print(f"Node {node} falls within a buffer zone. of {paths}")

                        unconnected_paths = return_paths_given_nodes(nodes, sp.paths_lst, sp.nodes_LUT,
                                                                     replace_nID=True, test='test')
                        plotter.plot_items(unconnected_paths)

                        # ax.scatter(nodes[0], nodes[1], s=120, marker='*', color='gold', zorder=3)


    plt.show()



def study_disconnected_comp():
    sp = doc.get_current_page()

    from itertools import chain

    print('start unconnected...')
    unconnected_primitives = [item for prim_k, prim_v in sp.primitives.items() if len(prim_v) < 3 for item in prim_v]
    unconnected_paths = return_paths_given_nodes(unconnected_primitives, sp.paths_lst, sp.nodes_LUT,
                                                 replace_nID=True, test='test')

    print('start connected...')
    connected_primitives = [item for prim_k, prim_v in sp.primitives.items() if len(prim_v) > 2 for item in prim_v]
    connected_paths = return_paths_given_nodes(connected_primitives, sp.paths_lst, sp.nodes_LUT,
                                               replace_nID=True)

    plt.figure()
    plt.imshow(sp.e_canvas)
    plotter.plot_items(unconnected_paths)
    plotter.plot_items(connected_paths)
    plt.show()

def study_line_fill_connection():
    sp = doc.get_current_page()

    plt.figure()
    plt.imshow(sp.e_canvas)

    items, nodes = [], []
    for path_id, path_value in sp.paths_lst.items():
        if path_value['path_type'] == 'f' and path_value['item_type'] == 'l':
            p1 = path_value['p1']
            p2 = path_value['p2']

            path_value['p1'] = sp.nodes_LUT[p1]
            path_value['p2'] = sp.nodes_LUT[p2]

            nodes.append(sp.nodes_LUT[p1])
            nodes.append(sp.nodes_LUT[p2])

            items.append(path_value)

    nodes = np.array(nodes)

    nodes_in_primitives = []
    for path_id, path_value in sp.paths_lst.items():
        # I used the paths here instead of the nodes_lut to have control over the path type.
        if not (path_value['path_type'] == 'f' and path_value['item_type'] == 'l'):
            p1 = path_value['p1']
            p2 = path_value['p2']
            p1_coords = sp.nodes_LUT[p1]
            p2_coords = sp.nodes_LUT[p2]

            for n_coords in nodes:
                if ((n_coords[0] - 1 < p1_coords[0] < n_coords[0] + 1
                     and n_coords[1] - 1 < p1_coords[1] < n_coords[1] + 1)
                        or (n_coords[0] - 1 < p2_coords[0] < n_coords[0] + 1
                            and n_coords[1] - 1 < p2_coords[1] < n_coords[1] + 1)):

                    if p1 not in nodes_in_primitives:
                        p_id, p_nodes = return_primitives_by_node(sp.primitives, p1)
                        nodes_in_primitives.extend(p_nodes)

    paths_overlapped = return_paths_given_nodes(nodes_in_primitives, sp.paths_lst, sp.nodes_LUT, replace_nID=True)
    items.extend(paths_overlapped)

    plotter.plot_items(items)
    plt.show()

    # exit()
        # exit()


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



