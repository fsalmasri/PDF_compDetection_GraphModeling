import math
from shapely.geometry import LineString


def bbox_to_polygon(bbx):
    from shapely.geometry import Polygon

    x0, y0, xn, yn = bbx
    # Define the corners of the bounding box
    corners = [(x0, y0), (xn, y0), (xn, yn), (x0, yn), (x0, y0)]

    # Create a Polygon
    polygon = Polygon(corners)

    return polygon

def is_point_inside_bbx(bbx, point):
    if point[0] > bbx[0] and point[0] < bbx[2]:
        if point[1] > bbx[1] and point[1] < bbx[3]:
            return True
    return False


    pass
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