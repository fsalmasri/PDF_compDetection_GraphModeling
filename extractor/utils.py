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


def keystoint(x):
    # return {int(k): v for k, v in x.items()}
    return {int(k) if k.lstrip('-').isdigit() else k: v for k, v in x.items()}


def return_path_given_nodes(nodes, path_lst):
    '''
    This function return path index (key value) in the path_list dictionary given a list of nodes.
    :param nodes: list of nodes to look for in the paths list
    :param path_lst: list of all paths to check.
    :return:
    '''
    paths_idx = []
    for n in nodes:
        for k, v in path_lst.items():
            if v['p1'] == n or v['p2'] == n:
                paths_idx.append(k)
    return np.unique(paths_idx)


def get_key_id(lst):
    # if the dictionary is empty return key = 1, otherwise return the last item key +1.
    if not lst.keys():
        key = 1
    else:
        key = list(lst.keys())[-1] + 1

    return key



def save_data(path_to_save, G, nodes_LUT, paths_lst, words_lst, blocks_lst, primitives):

    print('Saving all Lists ...')
    Path(f"{path_to_save}").mkdir(parents=True, exist_ok=True)

    nx.write_graphml_lxml(G, f"{path_to_save}/graph.graphml")
    # pickle.dump(G, open(f'{path_to_save}/graph.xml', 'wb'))
    #
    #
    # nx.write_gpickle(f'{path_to_save}/graph.txt')

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


def prepare_loaded_G(G):
    nodes_list = [int(x) for x in G.nodes]
    G.nodes = nodes_list
    edges_list = [(int(x), int(y)) for x, y in G.edges]

    G.remove_edges_from(list(G.edges()))
    G.add_edges_from(edges_list)

    return G

def load_data(path_to_save):

    G = nx.read_graphml(f"{path_to_save}/graph.graphml")
    # G = pickle.load(open(f'{path_to_save}/graph.txt'))
    G = prepare_loaded_G(G)



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

    # primitives = None

    return G, nodes_LUT, paths_lst, words_lst, blocks_lst, primitives


def get_bezier_cPoints(nodes, num_points=10):
    curve = bezier.Curve(nodes, degree=3)
    curve_points = curve.evaluate_multi(np.linspace(0, 1, num_points))
    return np.array(curve_points).T