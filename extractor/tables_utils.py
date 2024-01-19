from . import doc

from shapely.geometry import LineString, Point


def clean_tables_by_prims(prims_to_remove):
    '''
    Remove all paths, nodes, primitives using primitives ID.
    :param prims_to_remove: list primitives ID
    :return:
    '''
    sp = doc.get_current_page()

    for k_prim in prims_to_remove:
        paths_to_remove = [k_path for k_path, v_path in sp.paths_lst.items() if v_path['p_id'] == k_prim]
        nodes_to_remove = sp.primitives[k_prim]

        if k_prim in sp.primitives:
            del sp.primitives[k_prim]
        for path_id in paths_to_remove:
            del sp.paths_lst[path_id]
        for node_id in nodes_to_remove:
            del sp.nodes_LUT[node_id]

def delete_update_tables(to_delete_lst, ref='paths'):
    sp = doc.get_current_page()

    if ref == 'paths':
        # Clean paths list
        to_keep_paths = {k_path: v_path for k_path, v_path in sp.paths_lst.items() if k_path not in to_delete_lst}
        to_delete_paths = {k_path: v_path for k_path, v_path in sp.paths_lst.items() if k_path in to_delete_lst}
        print(f'duplicates paths to delete {len(to_delete_paths)}, Remaining {len(to_keep_paths)}')

        sp.paths_lst = to_keep_paths.copy()

        # Clean nodes list
        to_delete_edges = [tuple([x['p1'], x['p2']]) for x in to_delete_paths.values()]
        to_delete_nodes = {item for tpl in to_delete_edges for item in tpl}
        to_keep_nodes = set([coord for x in to_keep_paths.values() for coord in [x['p1'], x['p2']]])
        remaining_nodes = list(to_delete_nodes - to_keep_nodes)

        if remaining_nodes:
            filtered_nodes = list(to_keep_nodes - to_delete_nodes)
            # //TODO delete them from node list
            # //TODO Clean Primitives


            raise NotImplementedError
        else:
            print(f'no nodes to delete')

        # Clean Graph
        sp.G.remove_edges_from(to_delete_edges)

        sp.save_data(61)


    else:
        raise NotImplementedError

def clean_duplicates_paths():
    sp = doc.get_current_page()

    seen_paths = set()
    to_delete_lst = []

    selected_paths = {k:v for k, v in sp.paths_lst.items() if v["item_type"] == "l" and v["path_type"] == "f"}

    for k_path, v_path in selected_paths.items():
        path_tuple = tuple((tuple(sp.nodes_LUT[v_path['p1']]), tuple(sp.nodes_LUT[v_path['p2']])))

        if path_tuple in seen_paths:
            to_delete_lst.append(k_path)
        else:
            seen_paths.add(path_tuple)

    if to_delete_lst:
        delete_update_tables(to_delete_lst, ref='paths')


def return_new_nodeID():
    sp = doc.get_current_page()
    nodes_keys_list = list(sp.nodes_LUT.keys())
    nodes_keys_list.sort()

    new_key = nodes_keys_list[-1] + 1
    sp.nodes_LUT[new_key] = []

    return new_key

def return_new_pathID():
    sp = doc.get_current_page()
    paths_keys_list = list(sp.paths_lst.keys())
    paths_keys_list.sort()

    new_key = paths_keys_list[-1] + 1
    sp.paths_lst[new_key] = {}

    return new_key

def return_new_primitiveID():
    sp = doc.get_current_page()
    prim_keys_list = list(sp.primitives.keys())
    prim_keys_list.sort()

    new_key = prim_keys_list[-1] + 1
    sp.primitives[new_key] = {}

    return new_key


def split_creat_intersected_paths(intersect, path_to_split, path_touching, type='edge'):
    sp = doc.get_current_page()

    intersect_node = None
    if type == 'edge':
        if intersect[1] == tuple(sp.nodes_LUT[path_touching[1]['p1']]):
            intersect_node = path_touching[1]['p1']
        elif intersect[1] == tuple(sp.nodes_LUT[path_touching[1]['p2']]):
            intersect_node = path_touching[1]['p2']
    elif type == 'cross':
        intersect_node = return_new_nodeID()
        sp.nodes_LUT[intersect_node] = [intersect[1][0], intersect[1][1]]


    path_sec1_id = return_new_pathID()
    path_sec2_id = return_new_pathID()


    sp.paths_lst[path_sec1_id] = {'p1': path_to_split[1]['p1'], 'p2': intersect_node, 'item_type': path_to_split[1]['item_type'],
                                  'path_type': path_to_split[1]['path_type'], 'p_id': path_to_split[1]['p_id']}

    sp.paths_lst[path_sec2_id] = {'p1': intersect_node, 'p2': path_to_split[1]['p2'], 'item_type': path_to_split[1]['item_type'],
                                  'path_type': path_to_split[1]['path_type'], 'p_id': path_to_split[1]['p_id']}


    return path_sec1_id, path_sec2_id


def correct_path_ending_by_IntersectionPoint(k_line, v_line, intersection):
    sp = doc.get_current_page()

    sel_path = [[k, v] for k, v in sp.paths_lst.items() if v['p_id'] == k_line][0]

    start_distance = intersection.distance(Point(v_line.coords[0]))
    end_distance = intersection.distance(Point(v_line.coords[-1]))

    if start_distance < end_distance:
        v_line = LineString([(intersection.x, intersection.y)] + list(v_line.coords[1:]))
        node_to_correct = sel_path[1]['p1']
        sp.nodes_LUT[node_to_correct] = [intersection.x, intersection.y]


    else:
        v_line = LineString(list(v_line.coords[:-1]) + [(intersection.x, intersection.y)])
        node_to_correct = sel_path[1]['p2']
        sp.nodes_LUT[node_to_correct] = [intersection.x, intersection.y]

    return v_line