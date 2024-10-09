from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import bezier
import numpy as np
from math import atan2, degrees
import math
from itertools import chain

from collections import defaultdict
import networkx as nx
import json

from shapely.geometry import LineString, Point, Polygon, MultiLineString, MultiPolygon
from shapely.ops import cascaded_union
from shapely.ops import polygonize
import shapely.plotting
from shapely.ops import unary_union

from . import doc
from . import plotter
from . import tables_utils

from .utils import return_primitives_by_node, return_pathsIDX_given_nodes, return_paths_given_nodes, return_paths_by_primID
from .utils import return_nodes_by_region, prepare_region, check_PointRange

from . import utils
from . import bbx_margin

from .PID_utils import (remove_duplicates,
                        bbox_to_polygon,
                        is_point_inside_polygon,
                        get_bounding_box_of_points,
                        create_graph_from_paths,
                        paths_to_polygon,
                        detect_self_loop_path,
                        detect_overlaped_rectangles,
                        split_bimodal_distribution,
                        find_the_closest_point_to_polygon,
                        detect_Adjacent_primes)

from .bbx_utils import adjust_bbx_margin


def correct_grouped_primes(save_LUTs, plot, tag):
    sp = doc.get_current_page()

    info = sp.page_info
    if tag in info and info[tag]:
        return 0

    if plot:
        fig, ax = plt.subplots()
        plt.imshow(sp.e_canvas)

    selected_prims = {k: v for k, v in sp.primitives.items() if k not in sp.grouped_prims}
    for k_prime, v_prime in selected_prims.items():
        nodes_coords = [sp.nodes_LUT[x] for x in v_prime]
        paths_lst = return_paths_given_nodes(k_prime, v_prime, sp.paths_lst, sp.nodes_LUT, replace_nID=False, lst=False)
        paths_lst_with_coords = return_paths_given_nodes(k_prime, v_prime, sp.paths_lst, sp.nodes_LUT, replace_nID=True,
                                                         lst=False)

        # detect selfLoop path and remove it.
        selfloop_path_ids_to_delete = detect_self_loop_path(paths_lst)

        if len(selfloop_path_ids_to_delete) > 0:
            for id_to_delete in selfloop_path_ids_to_delete:
                del paths_lst[id_to_delete]
                del sp.paths_lst[id_to_delete]
                del paths_lst_with_coords[id_to_delete]

        new_paths_lst, to_delete = (
            detect_Adjacent_primes(paths_lst, highest_key=max(sp.primitives.keys())))

        # if the returned new_paths is bigger than one, means there was a split.
        if len(new_paths_lst) > 1:

            # update p_id in paths in the returned new_paths
            for p, v in new_paths_lst.items():
                for t, tt in v.items():
                    tt['p_id'] = p
                    # Update sp.paths_lst with new path value
                    sp.paths_lst[t] = tt

            # delete primes and its paths
            if len(to_delete) > 0:
                for k_delete in to_delete:
                    for p_delete in new_paths_lst[k_delete].keys():
                        del sp.paths_lst[p_delete]
                        del paths_lst_with_coords[p_delete]

                    del new_paths_lst[k_delete]

            del sp.primitives[k_prime]

            for k_nprime, v_nprime in new_paths_lst.items():
                nodes = [xv['p1'] for xk, xv in v_nprime.items()]
                sp.primitives[k_nprime] = nodes

    if save_LUTs:
        sp.save_primitives()
        sp.save_paths_lst()

        sp.page_info[tag] = True
        sp.save_info()


    sp.load_primitives()
    sp.load_paths_lst()

    if plot:
        selected_prims = {k: v for k, v in sp.primitives.items() if k not in sp.grouped_prims}
        for k_prime, v_prime in selected_prims.items():
            paths = return_paths_given_nodes(k_prime, v_prime, sp.paths_lst, sp.nodes_LUT, replace_nID=True)
            plotter.plot_items(paths, coloring='group')

        plt.show()

def detect_connections(save_LUTs, plot):
    sp = doc.get_current_page()

    if plot:
        fig, ax = plt.subplots()
        plt.imshow(sp.e_canvas)

    selected_prims = {k: v for k, v in sp.primitives.items() if k not in sp.grouped_prims}
    LCs = {k: v for k, v in sp.grouped_prims.items() if v['cls'] == "LC_input" or v['cls'] == "LC"}

    buffer_distance = 1

    # Build temporary dictionary of polygons from the selected primes to avoid repeating this procedure in next loop.
    tmp_selc_poly = {}
    for k_prime, v_prime in selected_prims.items():
        paths_lst_with_coords = return_paths_given_nodes(k_prime, v_prime, sp.paths_lst, sp.nodes_LUT, replace_nID=True,
                                                         lst=False)
        prime_poly, is_closed = paths_to_polygon(paths_lst_with_coords)
        if prime_poly is not None and not is_closed:
            buffered_con_poly = prime_poly.buffer(buffer_distance)
            tmp_selc_poly[k_prime] = buffered_con_poly


    founded_intersect = {}
    for k_prime, v_prime in tmp_selc_poly.items():

        for k_LC, v_LC in LCs.items():
            lc_bbx_poly = bbox_to_polygon(v_LC['bbx'])
            buffered_LC_poly = lc_bbx_poly.buffer(buffer_distance)

            if plot:
                x_buffered, y_buffered = buffered_LC_poly.exterior.xy
                plt.plot(x_buffered, y_buffered, color='red', linestyle='--',
                         label=f'Buffered Polygon (+{buffer_distance})')

            if buffered_LC_poly.intersects(v_prime):
                points_list = [sp.nodes_LUT[x] for x in selected_prims[k_prime]]
                intersect_coords, idx = find_the_closest_point_to_polygon(lc_bbx_poly, points_list)
                closest_node = selected_prims[k_prime][idx]

                if k_prime not in founded_intersect:
                    founded_intersect[k_prime] = {'nodes': selected_prims[k_prime], 'bbx': list(v_prime.bounds),
                                                  "cls": "con", "intersect_node": [closest_node], "intersect_coords": [intersect_coords]}
                else:
                    founded_intersect[k_prime]['intersect_node'].append(closest_node)
                    founded_intersect[k_prime]['intersect_coords'].append(intersect_coords)


    for k, v in founded_intersect.items():
        if len(v['intersect_node']) == 2:
            sp.grouped_prims[k] = v

            if plot:
                paths = return_paths_given_nodes(k, v['nodes'], sp.paths_lst, sp.nodes_LUT, replace_nID=True)
                plotter.plot_items(paths, coloring='group')


    if plot:
        plt.show()

    if save_LUTs:
        sp.save_grouped_prims()


def detect_LC_connectors(save_LUTs, plot, tag):
    sp = doc.get_current_page()

    info = sp.page_info
    if tag in info and info[tag]:
        return 0

    if plot:
        fig, ax = plt.subplots()
        plt.imshow(sp.e_canvas)

    selected_prims = {k: v for k, v in sp.primitives.items() if k not in sp.grouped_prims}

    for k_prime, v_prime in selected_prims.items():

        paths_lst_with_coords = return_paths_given_nodes(k_prime, v_prime, sp.paths_lst, sp.nodes_LUT, replace_nID=True,
                                                         lst=False)
        polygon, is_closed = paths_to_polygon(paths_lst_with_coords)
        if is_closed and len(v_prime) == 36:
            v_bbx = polygon.bounds
            v_bbx = adjust_bbx_margin(v_bbx, bbx_margin)
            sp.grouped_prims[k_prime] = {"nodes": v_prime, 'bbx': v_bbx, "cls": "LC_con"}
            if plot:
                paths = return_paths_given_nodes(k_prime, v_prime, sp.paths_lst, sp.nodes_LUT, replace_nID=True)
                plotter.plot_items(paths, coloring='group')

    if plot:
        plt.show()

    if save_LUTs:
        sp.save_grouped_prims()

        sp.page_info[tag] = True
        sp.save_info()

def detect_LC_rectangles(save_LUTs, plot, tag):
    sp = doc.get_current_page()
    parea = sp.ph * sp.pw

    info = sp.page_info
    if tag in info and info[tag]:
        return 0

    if plot:
        fig, ax = plt.subplots()
        plt.imshow(sp.e_canvas)

    selected_prims = {k: v for k, v in sp.primitives.items() if k not in sp.grouped_prims}

    tmp_polygons = {}
    for k_prime, v_prime in selected_prims.items():
        paths_lst_with_coords = return_paths_given_nodes(k_prime, v_prime, sp.paths_lst, sp.nodes_LUT, replace_nID=True,
                                                         lst=False)

        if len(paths_lst_with_coords) > 3:
            polygon, is_closed = paths_to_polygon(paths_lst_with_coords)
            if polygon is not None and is_closed:
                area = polygon.area / parea
                if area > 0.0002 and len(paths_lst_with_coords) < 15:
                    v_bbx = polygon.bounds
                    v_bbx = adjust_bbx_margin(v_bbx, bbx_margin)
                    tmp_polygons[k_prime] = {"nodes": v_prime, 'polygon': polygon, 'bbx': list(v_bbx), "cls": "LC", 'area': area}

    # # 1440
    # # 1445
    # def get_coords(polygon):
    #     x, y = polygon.exterior.xy
    #     return x, y
    #
    # x1, y1 = get_coords(tmp_polygons[1440]['polygon'])
    # x2, y2 = get_coords(tmp_polygons[1445]['polygon'])
    #
    # # Step 3: Plot the polygons
    # fig, ax = plt.subplots()
    # plt.imshow(sp.e_canvas)
    #
    # plt.plot(x1, y1, label='Polygon 1', color='blue')
    # plt.plot(x2, y2, label='Polygon 2', color='green')
    #
    # # Optional: Add labels, legend, and grid
    # plt.fill(x1, y1, alpha=0.5, color='blue')
    # plt.fill(x2, y2, alpha=0.5, color='green')
    #
    #
    # plt.show()
    #
    # exit()
    threshold = 1.1
    touching_or_inside_pairs  = []
    for poly_key1, v in tmp_polygons.items():
        polygon1 = v['polygon']
        area1 = v['area']
        for poly_key2, v in tmp_polygons.items():
            polygon2 = v['polygon']
            area2 = v['area']
            if poly_key1 != poly_key2:
                # print(poly_key1, poly_key2)
                if (polygon1.touches(polygon2) or polygon2.touches(polygon1) or
                        polygon1.within(polygon2) or polygon2.within(polygon1)):
                    if area2 > (area1 * threshold) or area1 > (area2 * threshold):
                        pair = tuple(sorted((poly_key1, poly_key2)))
                        if pair not in touching_or_inside_pairs:
                            touching_or_inside_pairs.append(pair)

    def merge_shared_pairssets(pairs):
        merged = []

        # Loop through the pairs and merge groups with shared items
        for pair in pairs:
            set_pair = set(pair)
            merged_with_existing = False

            # Check if the current pair shares any polygon with an existing group
            for group in merged:
                if not set_pair.isdisjoint(group):  # If they share any polygon
                    group.update(set_pair)  # Merge the pair into the group
                    merged_with_existing = True
                    break

            # If no existing group contains any polygon from the current pair, add a new group
            if not merged_with_existing:
                merged.append(set_pair)

        # Optional: Convert sets to sorted lists for easier readability
        return [sorted(group) for group in merged]

    # TODO we need to check if there is any shared polygon id between pairs, we need to merge them.
    merged_pairs = merge_shared_pairssets(touching_or_inside_pairs)


    merged_tmp_polygons = {}
    visited_pairs = set()
    for poly_key1, v in tmp_polygons.items():

        if poly_key1 not in visited_pairs:
            connected_polys = [x for x in merged_pairs if poly_key1 in x]
            if len(connected_polys) > 0:
                visited_pairs.update(connected_polys[0])
                merged_nodes = list(chain.from_iterable([tmp_polygons[x]['nodes'] for x in connected_polys[0]]))
                polys = [tmp_polygons[x]['polygon'] for x in connected_polys[0]]
                merged_polygon = unary_union(polys)
                v_bbx = merged_polygon.bounds
                v_bbx = adjust_bbx_margin(v_bbx, bbx_margin)
                new_v = {"nodes": merged_nodes, 'bbx': list(v_bbx), "cls": "LC", 'area': merged_polygon.area, 'p_ids':connected_polys[0]}
                merged_tmp_polygons[poly_key1] = new_v
            else:
                v['cls'] = "LC_input"
                merged_tmp_polygons[poly_key1] = v

    for k_prime, v_prime in merged_tmp_polygons.items():
        if 'polygon' in v_prime:
            del v_prime['polygon']

        sp.grouped_prims[k_prime] = v_prime

        if plot:
            if 'p_ids' in v_prime:
                paths = return_paths_given_nodes(v_prime['p_ids'], v_prime['nodes'], sp.paths_lst, sp.nodes_LUT,
                                                replace_nID=True)
                plotter.plot_items(paths, coloring='group')

            else:
                paths = return_paths_given_nodes(k_prime, selected_prims[k_prime], sp.paths_lst, sp.nodes_LUT,
                                                 replace_nID=True)
                plotter.plot_items(paths, coloring='test')
                # plt.text(paths[0]['p1'][0], paths[0]['p1'][1], k_prime, c='white', fontsize='small')
                # x0, y0, xn, yn = v_prime['bbx']
                # width = xn - x0
                # height = yn - y0
                # rect = patches.Rectangle((x0, y0), width, height,
                #                          linewidth=2, linestyle='dashed', edgecolor='r', facecolor='none')
                # ax.add_patch(rect)

    if plot:
        plt.show()

    if save_LUTs:
        sp.save_grouped_prims()

        sp.page_info[tag] = True
        sp.save_info()


# def detect_LC_rectangles2(save_LUTs, plot):
#     sp = doc.get_current_page()
#     parea = sp.ph * sp.pw
#
#     if plot:
#         fig, ax = plt.subplots()
#         plt.imshow(sp.e_canvas)
#
#     selected_prims = {k: v for k, v in sp.primitives.items() if k not in sp.grouped_prims}
#
#     temp_grouped_primes = {}
#     temp_LC_areas= []
#     for k_prime, v_prime in selected_prims.items():
#
#         nodes_coords = [sp.nodes_LUT[x] for x in v_prime]
#         paths_lst = return_paths_given_nodes(v_prime, sp.paths_lst, sp.nodes_LUT, replace_nID=False, lst=False)
#         paths_lst_with_coords = return_paths_given_nodes(v_prime, sp.paths_lst, sp.nodes_LUT, replace_nID=True, lst=False)
#
#
#         polygon, is_closed = paths_to_polygon(paths_lst_with_coords)
#         if polygon is not None and is_closed:
#             area = polygon.area / parea
#         else:
#             area = 0
#
#
#         if area > 0.0002 and len(paths_lst) < 15 and is_closed:
#             temp_LC_areas.append(area)
#             v_bbx = polygon.bounds
#             v_bbx= adjust_bbx_margin(v_bbx, bbx_margin)
#             temp_grouped_primes[k_prime] = {"nodes": v_prime, 'bbx': list(v_bbx), "cls": "LC", 'area': area}
#
#             if plot:
#                 paths = return_paths_given_nodes(v_prime, sp.paths_lst, sp.nodes_LUT, replace_nID=True)
#                 plotter.plot_items(paths, coloring='group')
#
#                 x0, y0, xn, yn = v_bbx
#                 width = xn - x0
#                 height = yn - y0
#                 rect = patches.Rectangle((x0, y0), width, height,
#                                          linewidth=2, linestyle = 'dashed', edgecolor='r', facecolor='none')
#                 ax.add_patch(rect)
#
#     # Split grouped_primes into two groups by area, LC and LC_INPUT
#     LC_INPUT, LC = split_bimodal_distribution(temp_grouped_primes)
#
#     for k_tmp, v_tmp in temp_grouped_primes.items():
#         if k_tmp in LC_INPUT:
#             sp.grouped_prims[k_tmp] = {"nodes": v_tmp['nodes'], 'bbx': v_tmp['bbx'], "cls": "LC_input"}
#         elif k_tmp in LC:
#             sp.grouped_prims[k_tmp] = {"nodes": v_tmp['nodes'], 'bbx': v_tmp['bbx'], "cls": "LC"}
#         else:
#             raise NotImplementedError("The data contains corrupted value.")
#
#     if plot:
#         plt.show()
#
#     if save_LUTs:
#         sp.save_grouped_prims()

def clean_text_by_OCR_bbxs(save_LUTs, plot, tag):
    '''

    Parse all connected points in primitives dictionary and check if they are contained in OCR bounding boxes
    if yes they mark them as char and save them in save_grouped_prims dictionary

    Args:
        save_LUTs: save save_grouped_prims dictionary
        plot: plot detected chars

    '''
    sp = doc.get_current_page()
    parea = sp.ph * sp.pw

    info = sp.page_info
    if tag in info and info[tag]:
        return 0

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
                paths = return_paths_given_nodes(k_prime, v_prime, sp.paths_lst, sp.nodes_LUT, replace_nID=True)
                plotter.plot_items(paths, coloring='group')

    if plot:
        plt.show()

    if save_LUTs:
        sp.save_grouped_prims()

        sp.page_info[tag] = True
        sp.save_info()




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

