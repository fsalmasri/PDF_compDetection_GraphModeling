
import numpy as np
import networkx as nx
from collections import defaultdict


from shapely.geometry import LineString, Point, Polygon

from . import doc
from . import plotter
from . import tables_utils

from .utils import return_paths_given_nodes
from .utils import check_PointRange


def is_vertical(line, threshold=10):
    angle = np.degrees(np.arctan2(line.xy[1][1] - line.xy[1][0], line.xy[0][1] - line.xy[0][0]))
    return abs(angle) > (90 - threshold) and abs(angle) < (90 + threshold)

def is_horizontal(line, threshold=10):
    angle = np.degrees(np.arctan2(line.xy[1][1] - line.xy[1][0], line.xy[0][1] - line.xy[0][0]))
    return abs(angle) < threshold or abs(angle) > (180 - threshold)



def Detect_unconnected_letters():
    sp = doc.get_current_page()

    # Biased values
    maximum_lines_length = 20.0
    percent_for_cross = 0.1

    vertical_dict = {}
    horizontal_dict = {}
    for k_iso_prime, v_iso_prime in sp.primitives.items():
        paths_isolated = return_paths_given_nodes(v_iso_prime, sp.paths_lst, sp.nodes_LUT, replace_nID=True, lst=False)

        paths_isolated_k, paths_isolated_v = next(iter(paths_isolated.items()))
        line = LineString([tuple(paths_isolated_v['p1']), tuple(paths_isolated_v['p2'])])

        if line.length < maximum_lines_length:  # Exclude zero-length lines
            if is_vertical(line):  # Check if line is approximately vertical
                vertical_dict[k_iso_prime] = line
                # vertical_lines.append(line)
            elif is_horizontal(line):  # Check if line is approximately horizontal
                horizontal_dict[k_iso_prime] = line
                # horizontal_lines.append(line)

    # print(f'{len(selected_paths)}, {len(horizontal_lines)}, {len(vertical_lines)}')
    print(f'horizontal lines: {len(horizontal_dict)}, vertical lines: {len(vertical_dict)}')

    # Find intersections between vertical and horizontal lines
    intersections = []
    for kv_line, vv_line in vertical_dict.items():
        flag = False
        v_sPoint = Point(vv_line.coords[0])
        v_ePoint = Point(vv_line.coords[-1])

        for kh_line, vh_line in horizontal_dict.items():
            h_sPoint = Point(vh_line.coords[0])
            h_ePoint = Point(vh_line.coords[-1])

            intersection = vv_line.intersection(vh_line)
            if intersection.is_empty is False and intersection.geom_type == 'Point':
                x, y = intersection.xy[0][0], intersection.xy[1][0]

                if intersection.equals(h_sPoint) or intersection.equals(h_ePoint):
                    intersection_type = 'horizental'
                    # [Type of intersection, coords of intersection, [path1, path2], path to split]
                    intersections.append(['edge', (x, y), [kv_line, kh_line], kv_line])
                elif intersection.equals(v_sPoint) or intersection.equals(v_ePoint):
                    intersection_type = 'vertical'
                    intersections.append(['edge', (x, y), [kv_line, kh_line], kh_line])
                else:
                    intersection_type = 'cross'

                    # cal the percentage of how far the intersection is along the vertical and horizontal lines
                    start_dis_along_v = intersection.distance(Point(vv_line.coords[0]))
                    end_dis_along_v = intersection.distance(Point(vv_line.coords[-1]))
                    percent_along_vertical = min([start_dis_along_v, end_dis_along_v]) / vv_line.length * 100

                    start_dis_along_h = intersection.distance(Point(vh_line.coords[0]))
                    end_dis_along_h = intersection.distance(Point(vh_line.coords[-1]))
                    percent_along_horizontal = min([start_dis_along_h, end_dis_along_h]) / vh_line.length * 100

                    if percent_along_vertical > 0.1 and percent_along_horizontal > 0.1:
                        # It is a cross, we have to add the two lines.
                        intersections.append(['cross', (x, y), [kv_line, kh_line]])

                    else:
                        if percent_along_vertical < percent_for_cross:
                            _ = tables_utils.correct_path_ending_by_IntersectionPoint(kv_line, vv_line,
                                                                                            intersection)
                            intersections.append(['edge', (x, y), [kv_line, kh_line], kh_line])

                        if percent_along_horizontal < percent_for_cross:
                            _ = tables_utils.correct_path_ending_by_IntersectionPoint(kh_line, vh_line,
                                                                                            intersection)
                            intersections.append(['edge', (x, y), [kv_line, kh_line], kv_line])


    new_paths_id = []
    print(f'found intersection: {len(intersections)}')
    for intersect in intersections:
        intersection_type = intersect[0]

        if intersection_type == 'edge':
            # get a copy of the selected paths
            path1 = [[k, v] for k, v in sp.paths_lst.items() if v['p_id'] == intersect[2][0]].copy()[0]
            path2 = [[k, v] for k, v in sp.paths_lst.items() if v['p_id'] == intersect[2][1]].copy()[0]

            path_to_change_id = intersect[3]
            if path_to_change_id == path1[1]['p_id']:
                del sp.paths_lst[path1[0]]
                print(f'path deleted {path1[0]}')
                path_sec1_id, path_sec2_id = (
                    tables_utils.split_creat_intersected_paths(intersect, path_to_split=path1, path_touching=path2))
                seconf_path_to_add = path2[0]

            elif path_to_change_id == path2[1]['p_id']:
                del sp.paths_lst[path2[0]]
                print(f'path deleted {path2[0]}')
                path_sec1_id, path_sec2_id = (
                    tables_utils.split_creat_intersected_paths(intersect, path_to_split=path2, path_touching=path1))
                seconf_path_to_add = path1[0]

            new_paths_id.append(path_sec1_id)
            new_paths_id.append(path_sec2_id)
            new_paths_id.append(seconf_path_to_add)

        else:
            path1 = [[k, v] for k, v in sp.paths_lst.items() if v['p_id'] == intersect[2][0]].copy()[0]
            path2 = [[k, v] for k, v in sp.paths_lst.items() if v['p_id'] == intersect[2][1]].copy()[0]
            del sp.paths_lst[path1[0]]
            del sp.paths_lst[path2[0]]
            print(f'path deleted {path1[0]} {path2[0]}')
            path_sec1_id, path_sec2_id = (
                tables_utils.split_creat_intersected_paths(intersect, path_to_split=path1, path_touching=path2, type='cross'))

            new_paths_id.append(path_sec1_id)
            new_paths_id.append(path_sec2_id)

            path_sec1_id, path_sec2_id = (
                tables_utils.split_creat_intersected_paths(intersect, path_to_split=path2, path_touching=path1, type='cross'))
            new_paths_id.append(path_sec1_id)
            new_paths_id.append(path_sec2_id)


    print(f'new_paths_id: {new_paths_id}')
    # exit()
    g = nx.Graph()
    for path_id in new_paths_id:
        path_v = sp.paths_lst[path_id].copy()
        g.add_node(path_v['p1'])
        g.add_node(path_v['p2'])
        g.add_edge(path_v['p1'], path_v['p2'])

        if path_v['p_id'] in sp.primitives:
            del sp.primitives[path_v['p_id']]

    connected_components = list(nx.connected_components(g))
    sp.update_primitives_tables(connected_components)


def remove_borders():
    sp = doc.get_current_page()

    # TODO define Global params for defining borders and tables in drawings
    exclude_borders_range = [[40, sp.pw], [40, sp.ph]]
    include_borders_range = [[1550, sp.pw], [1660, sp.ph]]

    prims_to_remove = set()
    for k_path, v_path in sp.paths_lst.items():
        p1_cond = check_PointRange(sp.nodes_LUT[v_path['p1']], exclude_borders_range)
        p2_cond = check_PointRange(sp.nodes_LUT[v_path['p2']], exclude_borders_range)
        if not (p1_cond and p2_cond):
            prims_to_remove.add(v_path['p_id'])
        p1_cond = check_PointRange(sp.nodes_LUT[v_path['p1']], include_borders_range)
        p2_cond = check_PointRange(sp.nodes_LUT[v_path['p2']], include_borders_range)
        if p1_cond and p2_cond:
            prims_to_remove.add(v_path['p_id'])

    tables_utils.clean_tables_by_prims(prims_to_remove)

def filter_bbxs(bounding_boxes):
    # Sort bounding boxes by area in descending order
    sorted_indices = sorted(range(len(bounding_boxes)), key=lambda i: bounding_boxes[i][2] * bounding_boxes[i][3],
                            reverse=True)
    sorted_bounding_boxes = np.array(bounding_boxes)[sorted_indices]

    # List to store indices of bounding boxes to remove
    to_remove_indices = []

    # List to store relationships (container, contained)
    relationships = defaultdict(list)

    # Iterate through the sorted bounding boxes
    for i, bb1 in enumerate(sorted_bounding_boxes):
        for j, bb2 in enumerate(sorted_bounding_boxes[i+1:]):
            # Check if bb2 is completely inside bb1
            if (
                bb2[0] >= bb1[0] and
                bb2[1] >= bb1[1] and
                (bb2[0] + bb2[2]) <= (bb1[0] + bb1[2]) and
                (bb2[1] + bb2[3]) <= (bb1[1] + bb1[3])
            ):
                # bb2 is completely inside bb1, mark for removal
                to_remove_indices.append(i + j + 1)
                relationships[i].append(i + j + 1)


    for i, bb in enumerate(sorted_bounding_boxes):
        if (i not in relationships) and (i not in to_remove_indices):
            relationships[i] = []

    return sorted_indices, relationships

def find_boundingBoxes(margin_percentage = 0.05):
    sp = doc.get_current_page()

    selected_primes = {prim_k: prim_v for prim_k, prim_v in sp.primitives.items() if len(prim_v) > 2}

    bbx_lst, kprims_lst = [], []
    for k_prim, v_prim in selected_primes.items():
        nodes_set = [tuple(sp.nodes_LUT[x]) for x in v_prim]

        path_geometry = LineString(nodes_set)
        bounding_box = path_geometry.bounds

        # Calculate area, height and width
        bounding_box_area = (bounding_box[2] - bounding_box[0]) * (bounding_box[3] - bounding_box[1])
        margin = (bounding_box_area ** 0.5) * margin_percentage

        # Expand the bounding box with the margin
        bounding_box_margin = (
            bounding_box[0] - margin,
            bounding_box[1] - margin,
            bounding_box[2] + margin,
            bounding_box[3] + margin
        )

        x = bounding_box_margin[0]
        y = bounding_box_margin[1]
        width = bounding_box_margin[2] - bounding_box_margin[0]
        height = bounding_box_margin[3] - bounding_box_margin[1]

        bbox = [x,y,width,height]

        if bounding_box_area < 46942:
            bbx_lst.append(bbox)
            kprims_lst.append(k_prim)


    sorted_indices, relationships = filter_bbxs(bbx_lst)
    sorted_bounding_boxes = [bbx_lst[i] for i in sorted_indices]
    sorted_kprims = [kprims_lst[i] for i in sorted_indices]

    for k_rel, v_rel in relationships.items():
        prim_id = sorted_kprims[k_rel]
        bbx = sorted_bounding_boxes[k_rel]
        sp.grouped_prims[prim_id] = {'nodes': sp.primitives[prim_id], 'bbx': bbx}

        vsub = sp.grouped_prims[prim_id]
        if v_rel:
            sub_dict = {}
            for v in v_rel:
                sub_prim_id = sorted_kprims[v]
                tmp_dict = {'nodes': sp.primitives[sub_prim_id], 'bbx': sorted_bounding_boxes[v]}
                sub_dict[sub_prim_id] = tmp_dict
            vsub['sub'] = sub_dict
        else:
            vsub['sub'] = None