from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import bezier
import numpy as np
from collections import defaultdict
import networkx as nx

from shapely.geometry import LineString, Point, Polygon, MultiLineString, MultiPolygon
from shapely.ops import cascaded_union
from shapely.ops import polygonize

from . import doc
from . import plotter
from . import tables_utils

from .utils import return_primitives_by_node, return_pathsIDX_given_nodes, return_paths_given_nodes, return_paths_by_primID
from .utils import return_nodes_by_region, prepare_region, check_PointRange





# from esig import tosig
# import esig

def normalize_coords(coords, image_width, image_height):
    x = coords[0] / image_width
    y = coords[1] / image_height

    return (x,y)

def study_clustering():
    histogram_length = 4

    sp = doc.get_current_page()

    image_width = sp.pw
    image_height = sp.ph

    signatures_dic = {}
    for k_gprims, v_gprims in sp.grouped_prims.items():
        sub_groups = v_gprims['sub']
        # main_nodes = [normalize_coords(sp.nodes_LUT[x], image_width, image_height) for x in v_gprims['nodes']]
        main_nodes = [sp.nodes_LUT[x] for x in v_gprims['nodes']]

        signature = esig.stream2sig(main_nodes, depth=3)
        signatures_dic[k_gprims] = [signature]

        if sub_groups is not None:
            for k_sub, v_sub in sub_groups.items():
                # sub_nodes = [normalize_coords(sp.nodes_LUT[x], image_width, image_height) for x in v_sub['nodes']]
                sub_nodes = [sp.nodes_LUT[x] for x in v_sub['nodes']]
                signature = esig.stream2sig(sub_nodes, depth=1)
                signatures_dic[k_gprims].append(signature)

        conc_signature = np.concatenate(signatures_dic[k_gprims], axis=0)

        histogram, _ = np.histogram(conc_signature, bins=histogram_length, density=True)

        # You may choose to normalize the histogram if needed
        # normalized_histogram = histogram / np.linalg.norm(histogram)
        signatures_dic[k_gprims] = histogram

    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import DBSCAN
    scaler = StandardScaler()


    all_hist = [v for k,v in signatures_dic.items()]
    all_hist = np.array(all_hist)

    histograms_array_scaled = scaler.fit_transform(all_hist)

    dbscan = DBSCAN(eps=.3, min_samples=5)  # You may need to adjust eps and min_samples based on your data
    cluster_labels = dbscan.fit_predict(histograms_array_scaled)

    print(cluster_labels)
    plt.scatter(histograms_array_scaled[:, 0], histograms_array_scaled[:, 1], c=cluster_labels, cmap='viridis')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('DBSCAN Clustering of Histograms')
    plt.show()

    for idx, (k_gprims, v_gprims) in enumerate(sp.grouped_prims.items()):
        v_gprims['class'] = int(cluster_labels[idx])





def shrink_line(line, shrink_factor):
    # Shrink the line by a factor
    start_point = line.interpolate(shrink_factor, normalized=True)
    end_point = line.interpolate(1 - shrink_factor, normalized=True)

    # Create a new LineString from the shrunken points
    shrunken_line = LineString([start_point, end_point])

    return shrunken_line


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









