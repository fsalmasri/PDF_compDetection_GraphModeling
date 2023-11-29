import numpy

def find_index_by_valueRange(LUT, rng: list):

    x, y = rng
    idx = []
    for k, v in LUT.items():
        if v[0] >= x[0] and v[0] <= x[1]:
            if v[1] >= y[0] and v[1] <= y[1]:
                idx.append(k)

    return idx


def return_values_by_Idx(LUT, nodes):
    pos = {}
    for n in nodes:
        pos[n] = LUT[n]

    return pos


def keystoint(x):
    return {int(k): v for k, v in x.items()}

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
            if n in v:
                paths_idx.append(k)
    return paths_idx
