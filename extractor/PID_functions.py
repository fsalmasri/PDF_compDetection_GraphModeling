from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import bezier
import numpy as np
from math import atan2, degrees
import math

from collections import defaultdict
import networkx as nx
import json

from shapely.geometry import LineString, Point, Polygon, MultiLineString, MultiPolygon
from shapely.ops import cascaded_union
from shapely.ops import polygonize
import shapely.plotting

from . import doc
from . import plotter
from . import tables_utils

from .utils import return_primitives_by_node, return_pathsIDX_given_nodes, return_paths_given_nodes, return_paths_by_primID
from .utils import return_nodes_by_region, prepare_region, check_PointRange

from . import utils

from .PID_utils import remove_duplicates, bbox_to_polygon, is_point_inside_polygon, get_bounding_box_of_points

def clean_text_by_OCR_bbxs(save_LUTs, plot):
    '''

    Parse all connected points in primitives dictionary and check if they are contained in OCR bounding boxes
    if yes they mark them as char and save them in save_grouped_prims dictionary

    Args:
        save_LUTs: save save_grouped_prims dictionary
        plot: plot detected chars

    '''
    sp = doc.get_current_page()
    parea = sp.ph * sp.pw

    if plot:
        fig, ax = plt.subplots()
        plt.imshow(sp.e_canvas)

    with open(f'{sp.pdf_saving_path}/OCRbox.json') as jf:
        OCR_dict = json.load(jf, object_hook=utils.keystoint)

    OCR_bbx = [[k[0][0]*sp.pw,k[0][1]*sp.ph,k[0][2]*sp.pw,k[0][3]*sp.ph] for v, k in OCR_dict.items()]
    OCR_bbx_polygon = [bbox_to_polygon(x) for x in OCR_bbx]

    selected_prims = {k: v for k, v in sp.primitives.items()}
    for k_prime, v_prime in selected_prims.items():
        coords = [sp.nodes_LUT[x] for x in v_prime]
        nodes_containing = [is_point_inside_polygon(OCR_bbx_polygon, x) for x in coords]
        bbx_id = nodes_containing[0][0]
        all_node_inside = all([elem[1] for elem in nodes_containing])

        if all_node_inside:
            v_bbx = get_bounding_box_of_points(coords)
            sp.grouped_prims[k_prime] = {"nodes": v_prime, 'bbx': v_bbx, "cls": "char", "OCR": OCR_dict[bbx_id][0][4]}

            if plot:
                paths = return_paths_given_nodes(v_prime, sp.paths_lst, sp.nodes_LUT, replace_nID=True)
                plotter.plot_items(paths, coloring='group')

    if plot:
        plt.show()

    if save_LUTs:
        sp.save_grouped_prims()




def filter_overlapped_polygons(polygons_list):
    '''
    Check if polygons overlap with each other and remove the small ones if founded.
    :param polygons_list:
    :return:
    '''
    marked_for_removal = set()

    for i, poly1 in enumerate(polygons_list):
        for j, poly2 in enumerate(polygons_list):

            buffer1 = poly1.buffer(0)
            buffer2 = poly2.buffer(0)

            if i != j:  # Avoid comparing the polygon with itself
                # print(i, buffer1.is_valid, j, buffer2.is_valid)
                if buffer1.intersects(buffer2): #and poly1.contains(poly2):

                    if poly1.area > poly2.area:
                        marked_for_removal.add(j)
                    else:
                        marked_for_removal.add(i)

    # print(list(marked_for_removal))
    # print(list(marked_to_keep))
    # exit()
    # Remove marked polygons from the list
    filtered_polygons = [poly for i, poly in enumerate(polygons_list) if i not in marked_for_removal]
    # filtered_polygons = [poly for i, poly in enumerate(polygons_list) if i in marked_to_keep]

    return filtered_polygons, marked_for_removal

def calculate_angles(multipolygon):
    """
    This function calculates all angles between consecutive segments in a Shapely multipolygon.

    Args:
      multipolygon: A Shapely multipolygon object.

    Returns:
      A list of angles in degrees for each linestring in the multipolygon.
    """

    mycoordslist = [list(x.exterior.coords) for x in multipolygon.geoms]
    mycoordslist2 = [list(x.interiors) for x in multipolygon.geoms]


    line_angles = []
    for coords in mycoordslist:
        prev_x, prev_y = coords[-2]
        curr_x, curr_y = coords[0]
        next_x, next_y = coords[1]
        angle = math.degrees(math.atan2(abs(next_y - curr_y), abs(next_x - curr_x)) -
                             math.atan2(abs(prev_y - curr_y), abs(prev_x - curr_x)))
        line_angles.append(angle)

        for i in range(1, len(coords)-1):
            prev_x, prev_y = coords[i-1]
            curr_x, curr_y = coords[i]
            next_x, next_y = coords[i+1]
            angle = math.degrees(math.atan2(abs(next_y - curr_y), abs(next_x - curr_x)) -
                                 math.atan2(abs(prev_y - curr_y), abs(prev_x - curr_x)))
            line_angles.append(angle)

    return [abs(ang) for ang in line_angles]


def find_polygons_in_paths_lst(paths):
    # # TODO maybe you change it to check the end to check a sequence of paths instead of closed shape.
    # # TODO then check if it is closed.

    multi_polygons = []
    path_geometries = []

    poly_idxs = []
    multipoly_idxs = []

    new_poly_flag = False
    l = LineString([tuple(paths[0]['p1']), tuple(paths[0]['p2'])])
    path_geometries.append(l)
    poly_idxs.append(0)

    start_point = l.coords[0]
    ending_point = l.coords[-1]

    for idx, path in enumerate(paths[1:]):
        l = LineString([tuple(path['p1']), tuple(path['p2'])])

        if new_poly_flag == True:
            start_point = l.coords[0]
            ending_point = l.coords[-1]
            path_geometries.append(l)
            poly_idxs.append(idx + 1)
            new_poly_flag = False

        else:
            if l.coords[0] == ending_point:
                ending_point = l.coords[-1]
                path_geometries.append(l)
                poly_idxs.append(idx+1)
                connected_flag = True
            else:
                connected_flag = False

            if l.coords[-1] == start_point and connected_flag:
                multi_polygons.append(path_geometries.copy())
                multipoly_idxs.append(poly_idxs)

                path_geometries = []
                poly_idxs = []
                new_poly_flag = True

    return multi_polygons, multipoly_idxs

def detect_rectangles():

    sp = doc.get_current_page()
    parea = sp.ph * sp.pw

    # fig, ax = plt.subplots()
    # plt.imshow(sp.e_canvas)

    selected_prims = {k:v for k, v in sp.primitives.items() if len(v) > 3}
    main_angles = []
    main_areas = []
    all_multipoly = []
    k_component = []
    nonCyclic_k_component = []
    for k_prime, v_prime in selected_prims.items():

        final_flag = False

        if True: #k_prime == 110:
            paths = return_paths_given_nodes(v_prime, sp.paths_lst, sp.nodes_LUT, replace_nID=True)
            paths_dic = return_paths_given_nodes(v_prime, sp.paths_lst, sp.nodes_LUT, replace_nID=False, lst=False)
            paths2 = return_paths_given_nodes(v_prime, sp.paths_lst, sp.nodes_LUT, replace_nID=False)

            # TODO Track removed paths and remove them from DB. Paths, Nodes, Primes.
            _, clean_paths_idx = remove_duplicates(paths)
            paths = [x for idx, x in enumerate(paths) if idx in clean_paths_idx]
            paths2 = [x for idx, x in enumerate(paths2) if idx in clean_paths_idx]

            G = nx.Graph()
            for p in paths2:
                G.add_edge(p['p1'], p['p2'])

            if list(nx.simple_cycles(G)):
                # This function is supposed to check closed polygons and add them to the list
                multi_polygons, multipoly_idxs = find_polygons_in_paths_lst(paths)

                poly = Polygon()
                polygons = []
                if len(multi_polygons) > 0:
                    duplicates_idxs = []
                    for midx, mploy in enumerate(multi_polygons):

                        # creat polygon using coords
                        mpoly_coords = []
                        mpoly_coords.append(mploy[0].coords[0])
                        for l in mploy:
                            mpoly_coords.append(l.coords[-1])

                        # find similar polygons for removal
                        cpoly = Polygon(mpoly_coords)
                        if cpoly.equals(poly):
                            duplicates_idxs.append(midx)
                        else:
                            poly = poly.union(cpoly)
                            polygons.append(cpoly)

                    cleaned_paths_idxs = [x for midx, x in enumerate(multipoly_idxs) if midx not in duplicates_idxs]
                    duplicates_paths_idxs = [x for midx, x in enumerate(multipoly_idxs) if midx in duplicates_idxs]
                    duplicates_paths_idxs = [item for sublist in duplicates_paths_idxs for item in sublist]

                    paths_dic_lst = list(paths_dic.items())

                    for p in duplicates_paths_idxs:
                        del sp.paths_lst[paths_dic_lst[p][0]]

                    multi_polygon = MultiPolygon(polygons)
                #
                #     # polygon characteristics
                #     angles = calculate_angles(multi_polygon)
                #     main_angles.append(angles)
                #     main_areas.append(multi_polygon.area / parea)
                #
                #     # TODO temporarly checking the area. must be changed to representative features.
                    if multi_polygon.area/parea > 0.0005: #np.mean(angles) > 20 and
                        all_multipoly.append(multi_polygon)
                        final_flag = True


        if final_flag:
            k_component.append(k_prime)
        else:
            nonCyclic_k_component.append(k_prime)

    final_poly, marked_for_removal = filter_overlapped_polygons(all_multipoly)
    filtered_k_components = [k for i, k in enumerate(k_component) if i not in marked_for_removal]



    fig, ax = plt.subplots()
    plt.imshow(sp.e_canvas)
    for k in filtered_k_components:
        v_prime = sp.primitives[k]
        paths = return_paths_given_nodes(v_prime, sp.paths_lst, sp.nodes_LUT, replace_nID=True)
        plotter.plot_items(paths, coloring='group')

        # plt.text(paths[0]['p1'][0], paths[0]['p1'][1], k, fontsize=10, color='white')

    # for k in nonCyclic_k_component:
    #     v_prime = sp.primitives[k]
    #     paths = return_paths_given_nodes(v_prime, sp.paths_lst, sp.nodes_LUT, replace_nID=True)
    #     plotter.plot_items(paths, coloring='test')
    #
    #     plt.text(paths[0]['p1'][0], paths[0]['p1'][1], k, fontsize=10, color='white')
    plt.show()

    return filtered_k_components

    # for f in final_poly:
    #     shapely.plotting.plot_polygon(f)
    # plt.show()




    #         # import shapely.plotting
    #         # shapely.plotting.plot_polygon(poly)
    #

    #         # if math.isclose(poly.minimum_rotated_rectangle.area, poly.area):
    #
    #         # if poly.boundary.is_ring:
    #             # print(poly.boundary.is_closed)
    #             # if poly.boundary.is_ring:
    #                 shapely.plotting.plot_polygon(poly)
    #                 # plotter.plot_items(paths, coloring='group')
    # plt.show()


def detect_rectangles2():

    sp = doc.get_current_page()
    parea = sp.ph * sp.pw

    # fig, ax = plt.subplots()
    # plt.imshow(sp.e_canvas)

    cyclic_k_component = []
    nonCyclic_k_component = []
    for k_prime, v_prime in sp.primitives.items():
        final_flag = False

        if k_prime == 41:
            paths = return_paths_given_nodes(v_prime, sp.paths_lst, sp.nodes_LUT, replace_nID=True)
            paths2 = return_paths_given_nodes(v_prime, sp.paths_lst, sp.nodes_LUT, replace_nID=False)

            _, clean_paths_idx = remove_duplicates(paths)
            print(clean_paths_idx)
            paths = [x for idx, x in enumerate(paths) if idx in clean_paths_idx]
            paths2 = [x for idx, x in enumerate(paths2) if idx in clean_paths_idx]


            # print(paths)
            # exit()
            G = nx.Graph()
            for idx, p in enumerate(paths2):
                G.add_node(p['p1'], pos=paths[idx]['p1'])
                G.add_node(p['p2'], pos=paths[idx]['p2'])
                G.add_edge(p['p1'], p['p2'])

            pos = nx.get_node_attributes(G, 'pos')

            # check if there is cycle
            if True: #list(nx.simple_cycles(G)):
                cyclic_k_component.append(k_prime)

                plt.figure()
                nx.draw(G, pos, with_labels=True)

        # import pickle
        # pickle.dump(G, open(f'graph_set/{k_prime}.pickle', 'wb'))

            fig, ax = plt.subplots()
            plt.imshow(sp.e_canvas)

            plotter.plot_items(paths, coloring='random')
            plt.text(paths[0]['p1'][0], paths[0]['p1'][1], k_prime, fontsize=10, color='white')

            plt.show()

    # for k in cyclic_k_component:
    #     v_prime = sp.primitives[k]
    #     paths = return_paths_given_nodes(v_prime, sp.paths_lst, sp.nodes_LUT, replace_nID=True)
    #     plotter.plot_items(paths, coloring='group')
    #     plt.text(paths[0]['p1'][0], paths[0]['p1'][1], k, fontsize=10, color='white')

    # for k in nonCyclic_k_component:
    #     v_prime = sp.primitives[k]
    #     paths = return_paths_given_nodes(v_prime, sp.paths_lst, sp.nodes_LUT, replace_nID=True)
    #     plotter.plot_items(paths, coloring='test')

    # plt.show()

    return 0

