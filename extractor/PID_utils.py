import math

import matplotlib.pyplot as plt
from shapely.geometry import LineString, Polygon, Point, MultiPoint
import networkx as nx
from sklearn.mixture import GaussianMixture
import numpy as np

from collections import defaultdict


def find_the_closest_point_to_polygon(polygon, points):
    closest_point = None
    min_distance = float('inf')
    closest_index = -1

    for i, point_coords in enumerate(points):
        point = Point(point_coords)

        distance = point.distance(polygon)

        if distance < min_distance:
            min_distance = distance
            closest_point = point_coords
            closest_index = i

    return closest_point, closest_index

def split_bimodal_distribution(data):
    """
    Splits a dictionary of data into two based on the bimodal distribution of the 'area' key.

    Parameters:
    - data_dict: Dictionary where each value is another dictionary containing an 'area' key

    Returns:
    - lower_dist: list of dictionary keys of the first group of data (lower mode)
    - upper_dist: list of dictionary keys of the second group of data (upper mode)
    """

    areas_dist = np.array([item['area'] for item in data.values()])

    # Fit a Gaussian Mixture Model with 2 components
    gmm = GaussianMixture(n_components=2)
    gmm.fit(areas_dist.reshape(-1, 1))
    # Get the means and covariances of the fitted distributions
    means = gmm.means_.flatten()
    # The threshold is typically the midpoint between the means
    threshold = np.mean(means)

    lower_dist = [k for k, v in data.items() if v['area'] < threshold]
    upper_dist = [k for k, v in data.items() if v['area'] >= threshold]

    return lower_dist, upper_dist


# Function to find the index of a tuple in the list
def find_index_of_pair(pairs_list, pair):
    try:
        return pairs_list.index(pair)  # Return the index directly
    except ValueError:
        return -1  # Return -1 if the pair is not found

def split_ltype_shared_edges(paths_lst, new_key):
    '''
    # TODO this function created issues when edges are nor ordered the same way in sequence. primes where cut into half.
    where there is ltype shapes that share the same edge, we split them into two primes.
    Args:
        paths_lst:
        new_key:

    Returns:

    '''
    sides = []
    for p in paths_lst.values():
        sides.append([p['p1'], p['p2']])

    # find duplicates pairs/side and set the founded duplicates.
    seen_pairs = set()
    duplicates = []
    for pair in sides:
        # Normalize the pair by sorting it
        normalized_pair = tuple(sorted(pair))

        # Check if the normalized pair is already in the seen set
        if normalized_pair in seen_pairs:
            duplicates.append(normalized_pair)
        else:
            seen_pairs.add(normalized_pair)

    if len(duplicates) > 0:
        duplicates_counter = [0] * len(duplicates)
        p_id = new_key  # set a new p_id value # TODO must return the p_id value to delete.

        new_primes = {}
        dic = {}
        for k, p in paths_lst.items():
            dic[k] = p
            new_primes[p_id] = dic

            pair = [p['p1'], p['p2']]
            normalized_pair = tuple(sorted(pair))
            d_idx = find_index_of_pair(duplicates, normalized_pair)

            if d_idx != -1:
                duplicates_counter[d_idx] += 1
                if duplicates_counter[d_idx] > 1:
                    p_id += 1
                    dic = {}

        return new_primes, True, p_id

    else:
        return paths_lst, False, None

def split_primes_by_closed_loop(paths_lst, new_key):
    '''
    Verify of quad is a composed quads. if yes split them by looping over nodes.
    Args:
        paths_lst: list of paths (dic)
        new_key: new kew for the primitives xml file.

    Returns: a dictionary of a dictionary of paths that represent a quad.

    '''
    if len(paths_lst) > 4:
        p_id = new_key
        new_primes = {}
        node = -1
        dic = {}
        for q, v in paths_lst.items():
            dic[q] = v
            new_primes[p_id] = dic
            if node == -1:
                node = v['p1']
            else:
                if node == v['p2']:
                    node = -1
                    p_id += 1
                    dic = {}

        return new_primes, True, p_id
    else:
        return paths_lst, False, None

def detect_Adjacent_primes(paths_nodes, highest_key):

    init_p_id = paths_nodes[list(paths_nodes.keys())[0]]['p_id']

    pTypes = [v['item_type'] for k, v in paths_nodes.items()]

    new_paths_nodes_lst = {}
    if 'l' in pTypes:
        lTypes = {k: v for k, v in paths_nodes.items() if v['item_type'] == 'l'}

        # new_lTypes, lflag, nhighest_key = split_ltype_shared_edges(lTypes, highest_key + 1)
        new_lTypes, lflag, nhighest_key = split_primes_by_closed_loop(lTypes, highest_key + 1)
        if lflag:
            highest_key = nhighest_key
            new_paths_nodes_lst = new_paths_nodes_lst | new_lTypes
        else:
            nadjusted_dic = {init_p_id: lTypes}
            new_paths_nodes_lst = new_paths_nodes_lst | nadjusted_dic


    if 'qu' in pTypes:
        quTypes = {k: v for k, v in paths_nodes.items() if v['item_type'] == 'qu'}
        new_quTypes, qflag, nhighest_key = split_primes_by_closed_loop(quTypes, highest_key + 1)

        if qflag:
            highest_key = nhighest_key
            new_paths_nodes_lst = new_paths_nodes_lst | new_quTypes
        else:
            nadjusted_dic = {init_p_id: quTypes}
            new_paths_nodes_lst = new_paths_nodes_lst | nadjusted_dic


    to_delete = []
    if 'l' in pTypes and 'qu' in pTypes:
        to_delete = detect_overlaped_rectangles(new_paths_nodes_lst, True)


    return new_paths_nodes_lst, to_delete



def detect_overlaped_rectangles(*args):
    if len(args) == 1:
        paths = args[0]

        pTypes = [v['item_type'] for k, v in paths.items()]
        if 'l' in pTypes and 'qu' in pTypes:
            lTypes = {k:v for k, v in paths.items() if v['item_type'] == 'l'}
            quTypes = {k:v for k, v in paths.items() if v['item_type'] == 'qu'}

            ltypes_points = [v['p1'] for k, v in lTypes.items()]
            ltypes_points.append(ltypes_points[0])
            ltype_polygon = Polygon(ltypes_points)

            qutypes_points = [v['p1'] for k, v in quTypes.items()]
            qutypes_points.append(qutypes_points[0])
            qutype_polygon = Polygon(qutypes_points)

            is_within = qutype_polygon.within(ltype_polygon)

            if is_within:
                return list(lTypes.keys())
            else:
                return None

    if len(args) == 2:
        paths = args[0]

        lTypes = {k: v for k, v in paths.items() for kk, l in v.items() if l['item_type'] == 'l'}
        quTypes = {k: v for k, v in paths.items() for kk, l in v.items() if l['item_type'] == 'qu'}

        founded = []
        for lk, lv in lTypes.items():
            lnodes = [xv['p1'] for x, xv in lv.items()]
            for quk, quv in quTypes.items():
                qunodes = [xv['p1'] for x, xv in quv.items()]
                if tuple(sorted(lnodes)) == tuple(sorted(qunodes)):
                    founded.append(lk)

        return founded


def create_graph_from_paths(paths):
    G = nx.Graph()  # Create an undirected graph
    for path in paths.values():
        p1 = path['p1']
        p2 = path['p2']
        G.add_edge(p1, p2)
    return G

def bbox_to_polygon(bbx):
    """
    Args:
        bbx: bounding box if the shape x0, y0, xn, yn

    Returns: return a polygon of the input bounding box
    """

    x0, y0, xn, yn = bbx
    # Define the corners of the bounding box
    corners = [(x0, y0), (xn, y0), (xn, yn), (x0, yn), (x0, y0)]

    return Polygon(corners)

def paths_to_polygon(paths):

    points = []
    for seg in paths.values():
        points.append(tuple(seg['p1']))
        points.append(tuple(seg['p2']))

    test = LineString(points)
    if len(points) >= 4:
        polygon = Polygon(points)
        return polygon, test.is_closed
    else:
        return None, False



def is_point_inside_bbx(bbx, point):
    if point[0] > bbx[0] and point[0] < bbx[2]:
        if point[1] > bbx[1] and point[1] < bbx[3]:
            return True
    return False

def is_point_inside_polygon(bbx_polygons, point):
    point_geom = Point(point)
    for idx, bbox_polygon in enumerate(bbx_polygons):
        if bbox_polygon.contains(point_geom):
            return idx, True
    return None, False

def detect_self_loop_path(paths):
    keys_to_delete = []
    for k, v in paths.items():
        if v['p1'] == v['p2']:
            keys_to_delete.append(k)

    return keys_to_delete

def get_bounding_box_of_points(points):
    """
    Compute the bounding box of a list of points.

    :param points: A list of (x, y) tuples representing points
    :return: A bounding box in the format [xmin, ymin, xmax, ymax]
    """
    if not points:
        raise ValueError("The list of points is empty")

    multipoint = MultiPoint(points)
    xmin, ymin, xmax, ymax = multipoint.bounds
    return [xmin, ymin, xmax, ymax]

def remove_duplicates(paths):

    clean_paths = []
    clean_paths_idx = []
    clean_paths_idx.append(0)
    clean_paths.append(paths[0])
    for idx, path in enumerate(paths[1:]):
        flag = True
        # for cpath in clean_paths:
            # if path['p1'] == cpath['p1'] and path['p2'] == cpath['p2']:
            #     # print('found it')
            #     flag = False
        if path['p1'] == path['p2']:
            flag = False

        if flag:
           clean_paths.append(path)
           clean_paths_idx.append(idx+1)

    return clean_paths, clean_paths_idx

def filter_overlapped_polygons(polygons_list):
    '''
    Check if polygons overlap with each other to remove the small ones if founded.
    :param polygons_list:
    :return: final polygons to keep, and the id of polygons to remove
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