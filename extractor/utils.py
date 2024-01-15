import json
from pathlib import Path
import bezier
import numpy as np
import pickle
import networkx as nx


def find_index_by_valueRange(LUT, rng: list):

    x, y = rng
    idx = []
    for k, v in LUT.items():
        if v[0] >= x[0] and v[0] <= x[1]:
            if v[1] >= y[0] and v[1] <= y[1]:
                idx.append(k)

    return idx

def look_in_txtBlocks_dict(LUT, textblocks):
    '''
    return nodes if their coords doesn't exist in any text blocks region.

    :param LUT: node list
    :param textblocks: textblocks list
    :return: nodes not included in text block regions
    '''
    nodes = []
    for n in LUT:
        break_outer = False
        for k, ltb in textblocks.items():
            for tb in ltb:
                if (tb[0]-1 <= LUT[n][0] <= tb[2]+1) and (tb[1]-1 <= LUT[n][1] <= tb[3]+1):
                    break_outer = True
                    break
            if break_outer:
                break

        if not break_outer:
            nodes.append(n)

    return nodes


def return_values_by_Idx(LUT, nodes):
    pos = {}
    for n in nodes:
        pos[n] = LUT[n]

    return pos


def return_pathsIDX_given_nodes(nodes, path_lst):
    '''
    This function return path index (key value) in the path_list dictionary given a list of nodes.
    :param nodes: list of nodes to look for in the paths list
    :param path_lst: list of all paths ID.
    :return:
    '''
    paths_idx = []
    for n in nodes:
        for k, v in path_lst.items():
            if v['p1'] == n or v['p2'] == n:
                paths_idx.append(k)
    return np.unique(paths_idx)

def return_paths_given_nodes(nodes, path_lst, nodes_LUT=None, replace_nID=True, test=None, lst = True):
    '''
    This function return path index (key value) in the path_list dictionary given a list of nodes.
    :param nodes: list of nodes to look for in the paths list
    :param path_lst: list of all paths. node ids are replaced with coordinates.
    :param replace_nID: Replace node ids with coordinates
    :param lst: boolean. if True return the output in list format. otherwise in dictionary.
    :return:
    '''

    if lst: paths = []
    else: paths = {}

    # this function is necessary to speed up the process when sending list of only nodes ids.
    # But when sending dictionary the set function removes the values in the dict and ket only the keys.
    if isinstance(nodes, list):
        nodes = set(nodes)

    for k, v in path_lst.items():
        if v['p1'] in nodes or v['p2'] in nodes:
            path = v.copy()
            if replace_nID:
                path['p1'] = nodes_LUT[v['p1']]
                path['p2'] = nodes_LUT[v['p2']]

            if test is not None:
                path['item_type'] = test
                path['path_type'] = test

            if lst: paths.append(path)
            else: paths[k] = path

    return paths

def return_primitives_by_node(primitives, n_id, lst: bool):
    '''

    :param primitives: primitives dictionary
    :param n_id: node id to look for in primitives dictionary
    :param lst: boolean. if True return the output in list format. otherwise in dictionary.
    :return: list of primitives id and its containing nodes.
    '''
    if lst:
        matching_primitives = [(prim_k, prim_v) for prim_k, prim_v in primitives.items() if n_id in prim_v]
        return matching_primitives[0][0], matching_primitives[0][1]
    else:
        return {prim_k: prim_v for prim_k, prim_v in primitives.items() if n_id in prim_v}

def return_primitives_by_pathLst(primitives, path_lst):
    '''
    :param primitives: primitives dictionary
    :param path_lst: Dictionary of paths to look for in primitives using p_id.
    :return: select dictionary elements by keys
    '''

    primitives_dict = {v_path['p_id']: primitives[v_path['p_id']] for k_path, v_path in path_lst.items() if
                       v_path['p_id'] in primitives}

    return primitives_dict

def return_nodes_by_region(nodes, x, y):
    '''
    x = [[],[]]
    y = [[],[]]
    :param nodes:
    :param x:
    :param y:
    :return:
    '''
    selected_nodes = {}
    for k, v in nodes.items():
        if x[0] < v[0] < x[1] and y[0] < v[1] < y[1]:
            selected_nodes[k] = v

    return selected_nodes

def prepare_region(nodes, path_lst, primitives, x, y):
    selected_nodes = return_nodes_by_region(nodes, x, y)
    selected_paths = return_paths_given_nodes(selected_nodes, path_lst, replace_nID=False, lst=False)
    selected_primitives = return_primitives_by_pathLst(primitives, selected_paths)

    return selected_nodes, selected_paths, selected_primitives

def check_PointRange(p, rng):
    '''
    rng=[[100,170],[460,560]]
    :param p:
    :param rng:
    :return:
    '''
    return p.x > rng[0][0] and p.x < rng[0][1] and p.y > rng[1][0] and p.y < rng[1][1]



def get_key_id(lst):
    # if the dictionary is empty return key = 1, otherwise return the last item key +1.
    if not lst.keys():
        key = 1
    else:
        key = list(lst.keys())[-1] + 1

    return key

def keystoint(x):
    # return {int(k): v for k, v in x.items()}
    return {int(k) if k.lstrip('-').isdigit() else k: v for k, v in x.items()}


def get_bezier_cPoints(nodes, num_points=10):
    curve = bezier.Curve(nodes, degree=3)
    curve_points = curve.evaluate_multi(np.linspace(0, 1, num_points))
    return np.array(curve_points).T

def prepare_loaded_G(G):
    nodes_list = [int(x) for x in G.nodes]
    G.nodes = nodes_list
    edges_list = [(int(x), int(y)) for x, y in G.edges]

    G.remove_edges_from(list(G.edges()))
    G.add_edges_from(edges_list)

    return G

def load_data(path_to_save):

    G = nx.read_graphml(f"{path_to_save}/graph.graphml", node_type=int, edge_key_type=int)
    # G = pickle.load(open(f'{path_to_save}/graph.txt'))
    # G = prepare_loaded_G(G)


    with open(f'{path_to_save}/nodes_LUT.json') as jf:
        nodes_LUT = json.load(jf, object_hook=keystoint)

    with open(f'{path_to_save}/paths_LUT.json') as jf:
        paths_lst = json.load(jf, object_hook=keystoint)

    with open(f'{path_to_save}/textBox.json') as jf:
        words_lst = json.load(jf, object_hook=keystoint)

    with open(f'{path_to_save}/blockBox.json') as jf:
        blocks_lst = json.load(jf, object_hook=keystoint)

    with open(f'{path_to_save}/primitives.json') as jf:
        primitives = json.load(jf, object_hook=keystoint)

    with open(f'{path_to_save}/filled_stroke.json') as jf:
        filled_stroke = json.load(jf, object_hook=keystoint)

    return G, nodes_LUT, paths_lst, words_lst, blocks_lst, primitives, filled_stroke

def save_svg(filename, svg):
    with open('{filename}.svg', 'w') as f:
        f.write(svg)

def save_data(path_to_save, G, nodes_LUT, paths_lst, words_lst, blocks_lst, primitives, filled_stroke):

    print('Saving all Lists ...')
    Path(f"{path_to_save}").mkdir(parents=True, exist_ok=True)

    nx.write_graphml_lxml(G, f"{path_to_save}/graph.graphml")
    # pickle.dump(G, open(f'{path_to_save}/graph.xml', 'wb'))

    with open(f"{path_to_save}/nodes_LUT.json", "w") as jf:
        json.dump(nodes_LUT, jf, indent=4)

    with open(f"{path_to_save}/paths_LUT.json", "w") as jf:
        json.dump(paths_lst, jf, indent=4)

    with open(f"{path_to_save}/textBox.json", "w") as jf:
        json.dump(words_lst, jf, indent=4)

    with open(f"{path_to_save}/blockBox.json", "w") as jf:
        json.dump(blocks_lst, jf, indent=4)

    with open(f"{path_to_save}/primitives.json", "w") as jf:
        json.dump(primitives, jf, indent=4)

    with open(f"{path_to_save}/filled_stroke.json", "w") as jf:
        json.dump(filled_stroke, jf, indent=4)

