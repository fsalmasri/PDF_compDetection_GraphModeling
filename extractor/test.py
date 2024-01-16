from collections import Counter
import matplotlib.pyplot as plt
import bezier
import numpy as np
from collections import defaultdict

from shapely.geometry import LineString, Point, Polygon, MultiLineString, MultiPolygon
from shapely.ops import cascaded_union
from shapely.ops import polygonize

from . import doc
from . import plotter
from . import tables_utils

from .utils import return_primitives_by_node, return_pathsIDX_given_nodes, return_paths_given_nodes
from .utils import return_nodes_by_region, prepare_region


def shrink_line(line, shrink_factor):
    # Shrink the line by a factor
    start_point = line.interpolate(shrink_factor, normalized=True)
    end_point = line.interpolate(1 - shrink_factor, normalized=True)

    # Create a new LineString from the shrunken points
    shrunken_line = LineString([start_point, end_point])

    return shrunken_line

def is_vertical(line, threshold=10):
    angle = np.degrees(np.arctan2(line.xy[1][1] - line.xy[1][0], line.xy[0][1] - line.xy[0][0]))
    return abs(angle) > (90 - threshold) and abs(angle) < (90 + threshold)

def is_horizontal(line, threshold=10):
    angle = np.degrees(np.arctan2(line.xy[1][1] - line.xy[1][0], line.xy[0][1] - line.xy[0][0]))
    return abs(angle) < threshold or abs(angle) > (180 - threshold)



def order_points_clockwise(points):
    import math
    # Calculate the centroid of the points
    centroid = Point(sum(x for x, y in points) / len(points), sum(y for x, y in points) / len(points))

    # Sort the points based on their polar angle relative to the centroid
    sorted_points = sorted(points, key=lambda p: math.atan2(p[1] - centroid.y, p[0] - centroid.x))

    return sorted_points

def clean_filled_strokes():
    sp = doc.get_current_page()

    # x = [15, 800]
    # y = [15, 800]
    #
    # selected_nodes, selected_paths, selected_primitives = (
    #     prepare_region(sp.nodes_LUT, sp.paths_lst, sp.primitives, x, y))

    fig, ax = plt.subplots()
    plt.imshow(sp.e_canvas)
    for k_prime, v_prime in sp.primitives.items():
        paths = return_paths_given_nodes(v_prime, sp.paths_lst, sp.nodes_LUT, replace_nID=True)
        plotter.plot_items(paths, coloring='group')

    for k_paths, v_paths in sp.filled_stroke.items():
        ordered_nodes = tuple([tuple(node) for x in v_paths for node in (x['p1'], x['p2'])])
        shape_polygon = Polygon(ordered_nodes)

        x, y = shape_polygon.exterior.xy
        plt.plot(x, y, 'r-', label='Polygon')

     # -------------------

    isolated_primes = {prim_k: prim_v for prim_k, prim_v in sp.primitives.items() if len(prim_v) < 3}

    fig, ax = plt.subplots()
    plt.imshow(sp.e_canvas)

    from shapely.prepared import prep
    from tqdm import tqdm

    for k_paths, v_paths in tqdm(sp.filled_stroke.items()):
        ordered_nodes = tuple([tuple(node) for x in v_paths for node in (x['p1'], x['p2'])])
        shape_polygon = Polygon(ordered_nodes)

        x, y = shape_polygon.exterior.xy
        plt.plot(x, y, 'r-', label='Polygon')

        # founded_paths = {}
        for k_iso_prim, v_iso_prim in isolated_primes.items():
            # paths = return_paths_given_nodes(v_iso_prim, sp.paths_lst, sp.nodes_LUT, replace_nID=True, lst=True)[0]
            # path_points = (tuple(paths['p1']), tuple(paths['p2']))
            path_points = (tuple(sp.nodes_LUT[v_iso_prim[0]]), tuple(sp.nodes_LUT[v_iso_prim[1]]))

            path_line = LineString(path_points)
            is_inside = path_line.within(shape_polygon)

            if is_inside:
                print(f'founded {k_paths} {k_iso_prim}')
                paths = return_paths_given_nodes(v_iso_prim, sp.paths_lst, sp.nodes_LUT, replace_nID=True, lst=True)[0]
                plotter.plot_items([paths], coloring='test')


    plt.show()


def Detect_unconnected_letters():
    sp = doc.get_current_page()

    x = [15, 200]
    y = [15, 200]

    # x = [15, 800]
    # y = [15, 800]
    #
    selected_nodes, selected_paths, selected_primitives = (
        prepare_region(sp.nodes_LUT, sp.paths_lst, sp.primitives, x, y))

    # fig, ax = plt.subplots()
    # plt.imshow(sp.e_canvas)
    # for k_prime, v_prime in selected_primitives.items():
    #     paths = return_paths_given_nodes(v_prime, sp.paths_lst, sp.nodes_LUT, replace_nID=True)
    #     plotter.plot_items(paths, coloring='standard')
    # isolated_primes = {prim_k: prim_v for prim_k, prim_v in selected_primitives.items() if len(prim_v) < 3}

    isolated_primes = {prim_k: prim_v for prim_k, prim_v in selected_primitives.items() if len(prim_v) < 3} #sp.primitives

    # vertical_lines = []
    # horizontal_lines = []

    fig, ax = plt.subplots()
    plt.imshow(sp.e_canvas)

    vertical_dict = {}
    horizontal_dict = {}
    for k_iso_prime, v_iso_prime in isolated_primes.items():
        paths_isolated = return_paths_given_nodes(v_iso_prime, sp.paths_lst, sp.nodes_LUT, replace_nID=True, lst=False)

        paths_isolated_k, paths_isolated_v = next(iter(paths_isolated.items()))
        line = LineString([tuple(paths_isolated_v['p1']), tuple(paths_isolated_v['p2'])])

        if line.length < 3.0:  # Exclude zero-length lines
            if is_vertical(line):  # Check if line is approximately vertical
                vertical_dict[k_iso_prime] = line
                # vertical_lines.append(line)
            elif is_horizontal(line):  # Check if line is approximately horizontal
                horizontal_dict[k_iso_prime] = line
                # horizontal_lines.append(line)

    # print(f'{len(selected_paths)}, {len(horizontal_lines)}, {len(vertical_lines)}')
    print(f'{len(horizontal_dict)}, {len(vertical_dict)}')

    # Find intersections between vertical and horizontal lines
    intersections = []
    for kv_line, vv_line in vertical_dict.items():
        flag = False
        v_sPoint = Point(vv_line.coords[0])
        v_ePoint = Point(vv_line.coords[-1])

        intersection_type = None
        for kh_line, vh_line in horizontal_dict.items():
            h_sPoint = Point(vh_line.coords[0])
            h_ePoint = Point(vh_line.coords[-1])

            intersection = vv_line.intersection(vh_line)
            if intersection.is_empty is False and intersection.geom_type == 'Point':
                x, y = intersection.xy[0][0], intersection.xy[1][0]

                if intersection.equals(h_sPoint) or intersection.equals(h_ePoint):
                    intersection_type = 'horizental'
                    # print(intersection_type)
                    intersections.append(['edge', (x, y), [kv_line, kh_line], kv_line])
                elif intersection.equals(v_sPoint) or intersection.equals(v_ePoint):
                    intersection_type = 'vertical'
                    # print(intersection_type)
                    intersections.append(['edge', (x, y), [kv_line, kh_line], kh_line])
                else:
                    # only in intersection we add a new node.
                    new_nodeID = tables_utils.return_new_nodeID()

                    # intersection_type = 'intersection'
                    # intersections.append(['edge', (x, y), [kv_line, kh_line], kv_line])

                    raise NotImplementedError

                x_shape, y_shape = vh_line.xy
                plt.plot(x_shape, y_shape, 'g-', label='Original Shape')
                flag = True
        if flag:
            x_shape, y_shape = vv_line.xy
            plt.plot(x_shape, y_shape, 'g-', label='Original Shape')

    # sp.nodes_LUT, sp.paths_lst, sp.primitives
    print(f'found intersection: {len(intersections)}')
    for intersect in intersections:
        print(intersect)
        intersection_type = intersect[0]

        if intersection_type == 'edge':
            path1_pid = intersect[2][0]
            path2_pid = intersect[2][1]

            # get a copy of the selected paths
            path1 = [[k, v] for k, v in sp.paths_lst.items() if v['p_id'] == path1_pid].copy()[0]
            path2 = [[k, v] for k, v in sp.paths_lst.items() if v['p_id'] == path2_pid].copy()[0]

            # get their keys
            path1_key = path1[0]
            path2_key = path2[0]

            # # Delete paths
            # del sp.paths_lst[path1_key]
            # del sp.paths_lst[path2_key]

            path_to_change_id = intersect[3]

            if path_to_change_id == path1[1]['p_id']:
                print('1')

                del sp.paths_lst[path1_key]

                intersect_node = None
                if intersect[1] == tuple(sp.nodes_LUT[path2[1]['p1']]):
                    intersect_node = path2[1]['p1']
                elif intersect[1] == tuple(sp.nodes_LUT[path2[1]['p2']]):
                    intersect_node = path2[1]['p2']

                # Generate new paths Ids
                path_sec1_id = tables_utils.return_new_pathID()
                path_sec2_id = tables_utils.return_new_pathID()
                path2_id = tables_utils.return_new_pathID()

                # Generate the new path two sections and a new other intersected path.
                path_sec1 = {path_sec1_id: {'p1': path1[1]['p1'], 'p2': intersect_node, 'item_type': path1[1]['item_type'],
                                            'path_type': path1[1]['path_type'], 'p_id': path1[1]['p_id']}}

                path_sec2 = {path_sec2_id: {'p1': intersect_node, 'p2': path1[1]['p2'], 'item_type': path1[1]['item_type'],
                                            'path_type': path1[1]['path_type'], 'p_id': path1[1]['p_id']}}

                print(path2)
                print(path_sec1)
                print(path_sec1)
                exit()
                paths_to_add = [path2, path_sec1, path_sec2]

                # # Merge the primitives into a new one.
                # prim_id = tables_utils.return_new_primitiveID()
                # nodes_to_add = set()
                # for path in paths_to_add:
                #     path_values = list(path.values())[0]
                #     nodes_to_add.add(path_values['p1'])
                #     nodes_to_add.add(path_values['p2'])
                #
                # nodes_to_add = list(nodes_to_add)

                exit()


                print(path2)
                print(path_sec1)
                print(path_sec2)

                exit()
            elif path_to_change_id == path2[1]['p_id']:
                print('2')


            else:
                raise NotImplementedError

            print(path_to_change_id)
            print(path1[0], path1[1])
            print(path2[0], path2[1])
            exit()

        # intersect_point_id = intersect[0]
        # intersect_point_coords = intersect[1]
        # path1_prim_id = intersect[2]
        # path2_prim_id = intersect[3]
        # primitive_id = intersect[4]


        # //TODO added the broken line nodes. still need to add the new line.
        # //TODO I think it is better if I correct the path list first.
        prim_nodes = sp.primitives[primitive_id]
        sp.primitives[primitive_id] = [prim_nodes[0], intersect_point_id, prim_nodes[1]]


        # --------------------

        exit()
    print(intersections)
    plt.show()
    exit()


def Clean_filling_strikes():
    sp = doc.get_current_page()

    # x = [120, 135]
    # y = [50, 75]

    x = [15, 100]
    y = [15, 100]

    shrink_factor = 0.2
    selected_nodes, selected_paths, selected_primitives = prepare_region(sp.nodes_LUT, sp.paths_lst, sp.primitives, x, y)
    #
    fig, ax = plt.subplots()
    plt.imshow(sp.e_canvas)
    for k_prime, v_prime in selected_primitives.items():
        paths = return_paths_given_nodes(v_prime, sp.paths_lst, sp.nodes_LUT, replace_nID=True)
        plotter.plot_items(paths, coloring='standard')
    # plt.show()
    # exit()

    fig, ax = plt.subplots()
    plt.imshow(sp.e_canvas)

    isolated_primes = {prim_k: prim_v for prim_k, prim_v in selected_primitives.items() if len(prim_v) < 3}
    connected_primes = {prim_k: prim_v for prim_k, prim_v in selected_primitives.items() if len(prim_v) > 3}

    print(f'Isolated primes: {len(isolated_primes)}  Connected primes: {len(connected_primes)}')

    all_deleted_keys = {}
    for k_con_prime, v_con_prime in connected_primes.items():
        shape_coords = [(x[0], x[1]) for k, x in sp.nodes_LUT.items() if k in v_con_prime]
        shape = Polygon(shape_coords)
        paths_connected = return_paths_given_nodes(v_con_prime, sp.paths_lst, sp.nodes_LUT, replace_nID=True)

        plotter.plot_items(paths_connected, coloring='group')
        # x_shape, y_shape = shape.exterior.xy
        # plt.plot(x_shape, y_shape, 'g-', label='Original Shape')

        # //TODO we should only check what is in the range instead of all the isolated comps.
        # Get the bounding box (range) in x, y coordinates
        # x_min, y_min, x_max, y_max = shape.bounds


        keys_to_delete = []
        for k_iso_prime, v_iso_prime in isolated_primes.items():
            paths_isolated = return_paths_given_nodes(v_iso_prime, sp.paths_lst, sp.nodes_LUT, replace_nID=True, lst=False)

            paths_isolated_k, paths_isolated_v = next(iter(paths_isolated.items()))
            line = LineString([tuple(paths_isolated_v['p1']), tuple(paths_isolated_v['p2'])])

            shrunken_line = shrink_line(line, shrink_factor)
            is_within_shape = shrunken_line.within(shape) #or shrunken_line.intersects(shape)

            if is_within_shape:
                # print(paths_isolated_k)
                keys_to_delete.append(k_iso_prime)
                # plotter.plot_items([paths_isolated_v], coloring='test')
            # else:
            #     plotter.plot_items([paths_isolated_v], coloring='group')
                # x_shrunken, y_shrunken = shrunken_line.xy
                # plt.plot(x_shrunken, y_shrunken, 'r-', label='Shrunken Line')


        for key in keys_to_delete:
            all_deleted_keys[key] = isolated_primes[key]
            del isolated_primes[key]

        print(f'Remaining isolated primes: {len(isolated_primes)}')


    for k_iso_prime, v_iso_prime in isolated_primes.items():
        paths_isolated = return_paths_given_nodes(v_iso_prime, sp.paths_lst, sp.nodes_LUT, replace_nID=True, lst=True)
        plotter.plot_items(paths_isolated, coloring='group')


    fig, ax = plt.subplots()
    plt.imshow(sp.e_canvas)
    for k_iso_prime, v_iso_prime in all_deleted_keys.items():
        paths_isolated = return_paths_given_nodes(v_iso_prime, sp.paths_lst, sp.nodes_LUT, replace_nID=True, lst=True)
        plotter.plot_items(paths_isolated, coloring='group')


    plt.show()




def plot_full_dwg(region = False, x=None, y=None):

    #// TODO should change it to use prepare_region() function.
    sp = doc.get_current_page()

    if region:
        x = [120,135]
        y = [50,75]
        selected_nodes = return_nodes_by_region(x, y)
        print(f'found {len(selected_nodes)}')

        canvas = sp.e_canvas #[x[0]:x[1], y[0]: y[1]]
    else:
        selected_nodes = sp.nodes_LUT
        canvas = sp.e_canvas

    fig, ax = plt.subplots()
    plt.imshow(canvas)

    for k, v in sp.paths_lst.items():
        if v['p1'] in selected_nodes or v['p2'] in selected_nodes:
            v['p1'] = sp.nodes_LUT[v['p1']]
            v['p2'] = sp.nodes_LUT[v['p2']]
            plotter.plot_items([v], coloring=False)

    plt.show()




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

    all_comp = [prim_v for prim_k, prim_v in sp.primitives.items()]
    connected_primitives = [prim_v for prim_k, prim_v in sp.primitives.items() if len(prim_v) > 2]
    unconnected_nodes = [prim_v for prim_k, prim_v in sp.primitives.items() if len(prim_v) < 3]
    node_buffer_distance = 0.1

    fig, ax = plt.subplots()
    plt.imshow(sp.e_canvas)

    for prim_idx, prim_v in enumerate(all_comp[:]):

        paths = return_paths_given_nodes(prim_v, sp.paths_lst, sp.nodes_LUT, replace_nID=True)

        flag = False
        if paths[0]['item_type'] == 'l':

            testing_comp = all_comp[:prim_idx] + all_comp[prim_idx + 1:]

            node_buffers = [Point(sp.nodes_LUT[node]).buffer(node_buffer_distance) for node in prim_v]

            for nodes in testing_comp:
                node = [sp.nodes_LUT[nodes[0]], sp.nodes_LUT[nodes[1]]]

                node_point_1 = Point(node[0])
                node_point_2 = Point(node[1])

                for buffer_zone in node_buffers:
                    if node_point_1.within(buffer_zone) or node_point_2.within(buffer_zone):
                        # print(f"Node {node} falls within a buffer zone. of {paths}")

                        unconnected_paths = return_paths_given_nodes(nodes, sp.paths_lst, sp.nodes_LUT,
                                                                     replace_nID=True, test='test')
                        plotter.plot_items(unconnected_paths)

                        flag = True

                        # ax.scatter(nodes[0], nodes[1], s=120, marker='*', color='gold', zorder=3)

            if flag:
                for buffered_path in node_buffers:
                    x, y = buffered_path.exterior.xy
                    ax.plot(x, y, color='pink')  # Adjust color and styling as needed

                plotter.plot_items(paths)

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


def check_dwg_item_type(dwg, type='c'):
    for item in dwg['items']:
        if item[0] == type: # or item[0] == 're' or item[0] == 'qu':
            return True

    return False


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



