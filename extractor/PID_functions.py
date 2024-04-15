from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import bezier
import numpy as np
from math import atan2, degrees
import math

from collections import defaultdict
import networkx as nx

from shapely.geometry import LineString, Point, Polygon, MultiLineString, MultiPolygon
from shapely.ops import cascaded_union
from shapely.ops import polygonize
import shapely.plotting

from . import doc
from . import plotter
from . import tables_utils

from .utils import return_primitives_by_node, return_pathsIDX_given_nodes, return_paths_given_nodes, return_paths_by_primID
from .utils import return_nodes_by_region, prepare_region, check_PointRange


def remove_duplicates(paths):

    clean_paths = []
    clean_paths.append(paths[0])
    for idx, path in enumerate(paths[1:]):
        flag = True
        for cpath in clean_paths:
            # if path['p1'] == cpath['p1'] and path['p2'] == cpath['p2']:
            #     # print('found it')
            #     flag = False
            if path['p1'] == path['p2']:
                flag = False

        if flag:
           clean_paths.append(path)

    return clean_paths

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

def detect_rectangles():
    from skimage.feature import hog
    from skimage import measure

    sp = doc.get_current_page()
    parea = sp.ph * sp.pw

    fig, ax = plt.subplots()
    plt.imshow(sp.e_canvas)

    main_angles = []
    main_areas = []
    all_multipoly = []
    for k_prime, v_prime in sp.primitives.items():

        if True: #k_prime == 41:
            paths = return_paths_given_nodes(v_prime, sp.paths_lst, sp.nodes_LUT, replace_nID=True)
            paths2 = return_paths_given_nodes(v_prime, sp.paths_lst, sp.nodes_LUT, replace_nID=False)
            # TODO Track removed paths and remove them from DB. Paths, Nodes, Primes.
            paths = remove_duplicates(paths)

            multi_polygons = []
            path_geometries = []

            G = nx.Graph()
            for p in paths2:
                G.add_edge(p['p1'], p['p2'])

            # check if there is cycle
            if list(nx.simple_cycles(G)):
                # This function is supposed to check closed polygons and add them to the list
                # TODO maybe you change it to check the end to check a sequence of paths instead of closed shape.
                # TODO then check if it is closed.
                new_poly_flag = False
                l = LineString([tuple(paths[0]['p1']), tuple(paths[0]['p2'])])
                path_geometries.append(l)
                start_point = l.coords[0]
                ending_point = l.coords[-1]

                for path in paths[1:]:
                    l = LineString([tuple(path['p1']), tuple(path['p2'])])

                    if new_poly_flag == True:
                        start_point = l.coords[0]
                        new_poly_flag = False

                    if l.coords[0] == ending_point:
                        ending_point = l.coords[-1]
                        path_geometries.append(l)
                        connected_flag = True
                    else:
                        connected_flag = False

                    if l.coords[-1] == start_point and connected_flag:
                        multi_polygons.append(path_geometries.copy())
                        path_geometries = []
                        new_poly_flag = True

                # print(len(multi_polygons), multi_polygons, '\n')

                poly = Polygon()
                polygons = []
                if len(multi_polygons) > 0:
                    for mploy in multi_polygons:

                        test = []
                        test.append(mploy[0].coords[0])
                        for l in mploy:
                            test.append(l.coords[-1])

                        if len(test) > 2:
                            cpoly = Polygon(test)

                            if cpoly.equals(poly):
                                new_poly_flag = True # TODO remove paths from DS.
                            else:
                                poly = poly.union(cpoly)
                                polygons.append(cpoly)

                    # exit()
                    multi_polygon = MultiPolygon(polygons)

                    angles = calculate_angles(multi_polygon)
                    main_angles.append(angles)

                    # plt.figure()
                    # plt.imshow(sp.e_canvas)

                    main_areas.append(multi_polygon.area / parea)
                    if True: #multi_polygon.area/parea > 0.0005: #np.mean(angles) > 20 and
                        # plotter.plot_items(paths, coloring='group')
                        # plt.text(paths[0]['p1'][0], paths[0]['p1'][1], k_prime, fontsize=10, color='white')
                        all_multipoly.append(multi_polygon)

    # plt.show()
                    # else:
                    #     plotter.plot_items(paths, coloring='test')

                    # fig, ax = plt.subplots()
                    # for geom in multi_polygon.geoms:
                    #     xs, ys = geom.exterior.xy
                    #     ax.fill(xs, ys, alpha=0.3, fc='r', ec='none')
                    #
                    # # shapely.plotting.plot_polygon(multi_polygon)

                    # plt.show()

    # exit()
    # sum_angles = [sum(x) for x in main_angles]
    # mean_angles = [np.mean(x) for x in main_angles]
    #
    # plt.figure()
    # plt.hist(sum_angles, bins=30)
    #
    # plt.figure()
    # plt.hist(mean_angles, bins=30)
    #
    # plt.figure()
    # plt.hist(main_areas, bins=30)
    # plt.show()


    #         # exit()
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

    def filter_polygons(polygons_list):
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

        return filtered_polygons

    final_poly = filter_polygons(all_multipoly)
    print(len(all_multipoly), len(final_poly))

    for f in final_poly:
        shapely.plotting.plot_polygon(f)
    plt.show()