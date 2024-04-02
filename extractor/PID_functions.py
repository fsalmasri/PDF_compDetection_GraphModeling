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
def detect_rectangles():
    from skimage.feature import hog
    from skimage import measure

    sp = doc.get_current_page()

    # fig, ax = plt.subplots()
    # plt.imshow(sp.e_canvas)

    for k_prime, v_prime in sp.primitives.items():
        paths = return_paths_given_nodes(v_prime, sp.paths_lst, sp.nodes_LUT, replace_nID=True)
        # TODO Track removed paths and remove them from DB. Paths, Nodes, Primes.
        paths = remove_duplicates(paths)

        # print(k_prime, v_prime)
        # for p in paths:
        #     print(p)
        # exit()

        multi_polygons = []
        path_geometries = []
        # path_geometries.append(LineString([tuple(paths[0]['p1']), tuple(paths[0]['p2'])]))

        flag = True
        for path in paths[0:]:

            l = LineString([tuple(path['p1']), tuple(path['p2'])])

            # TODO maybe you change it to check the end to check a sequence of paths instead of closed shape.
            if flag == True:
                start_point = l.coords[0]
                flag = False

            path_geometries.append(l)
            if l.coords[-1] == start_point:
                multi_polygons.append(path_geometries.copy())
                path_geometries = []
                flag = True


        print(len(multi_polygons), multi_polygons, '\n')

        poly = Polygon()
        polygons = []
        if len(multi_polygons) > 0:

            for mploy in multi_polygons:

                test = []
                test.append(mploy[0].coords[0])
                for l in mploy:
                    test.append(l.coords[-1])

                # print(test)

                if len(test) > 2:
                    cpoly = Polygon(test)

                    if cpoly.equals(poly):
                        flag = True # TODO remove paths from DS.
                    else:
                        poly = poly.union(cpoly)
                        polygons.append(cpoly)

            multi_polygon = MultiPolygon(polygons)

            import shapely.plotting as shplt

            print(multi_polygon.boundary)
            plt.figure()
            plt.imshow(sp.e_canvas)
            plotter.plot_items(paths, coloring='group')

            fig, ax = plt.subplots()
            for geom in multi_polygon.geoms:
                xs, ys = geom.exterior.xy
                ax.fill(xs, ys, alpha=0.3, fc='r', ec='none')

            # shapely.plotting.plot_polygon(multi_polygon)
            plt.show()
            # exit()

    #         # print(poly.area)
    #
    #         # print(path_geometries)
    #         # print(poly)
    #         #
    #         # print(poly.area)
    #         # print(poly.minimum_rotated_rectangle.area)
    #
    #         # exit()
    #         # import shapely.plotting
    #         # shapely.plotting.plot_polygon(poly)
    #
    #         import math
    #
    #
    #         # if math.isclose(poly.minimum_rotated_rectangle.area, poly.area):
    #
    #
    #         if poly.area > 200:
    #
    #             # print(k_prime)
    #             if k_prime == 75:
    #                 print(poly.boundary)
    #                 print(poly.is_closed, poly.is_ring)
    #
    #         # if poly.boundary.is_ring:
    #             # print(poly.boundary.is_closed)
    #             # exit()
    #
    #             # if poly.boundary.is_ring:
    #             # print(poly.boundary.coords[0], poly.boundary.coords[-1], poly.boundary.is_closed)
    #             # exit()
    #             # if poly.boundary.coords[0] == poly.boundary.coords[-1]:
    #             # fig, ax = plt.subplots()
    #             # plt.imshow(sp.e_canvas)
    #
    #                 shapely.plotting.plot_polygon(poly)
    #                 # plotter.plot_items(paths, coloring='group')
    #             # plt.show()
    # plt.show()


        # for f in path_geometries:
        #     plt.plot(*f.xy)
        # # plotter.plot_items(paths, coloring='random')
