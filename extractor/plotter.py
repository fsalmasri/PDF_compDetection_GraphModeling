import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import networkx as nx
import random

from . import doc

from .utils import return_nodes_by_region
from .utils import return_paths_given_nodes

import json
from .utils import keystoint

color_mapping = {'s': 'white', 'f': 'yellow', 'fs': 'blue',
                 'qu': 'purple', 're': 'red', 'c': 'orange', 'test': 'yellow'}

def plot_items(items, coloring = 'standard'):

    path_color = get_colors(random.randint(0, 128))
    for path in items:
        if coloring == 'standard':
            path_color = color_mapping[path['path_type']] if path['item_type'] == 'l' else color_mapping[
                path['item_type']]
            plt.plot([path['p1'][0], path['p2'][0]], [path['p1'][1], path['p2'][1]], c=path_color)

        elif coloring == 'random':
            plt.plot([path['p1'][0], path['p2'][0]], [path['p1'][1], path['p2'][1]])

        elif coloring == 'group':
            plt.plot([path['p1'][0], path['p2'][0]], [path['p1'][1], path['p2'][1]], c=path_color)

        elif coloring == 'test':
            plt.plot([path['p1'][0], path['p2'][0]], [path['p1'][1], path['p2'][1]], c=color_mapping['test'])



def plot_graph_nx(g= None):

    sp = doc.get_current_page()
    if g is None:
        g = sp.G

    pos = {}
    for n in g.nodes:
        pos[n] = np.array(sp.nodes_LUT[n])

    fig, ax = plt.subplots()
    nx.draw(g, pos, with_labels=True, node_size=200, node_color='lightblue',
            font_size=7, font_color='black', font_weight='bold', width=2, ax=ax)

    plt.gca().invert_yaxis()
    plt.axis('on')
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    plt.show()



def plot_full_dwg(region=False, paths=True, connected_com= True, OCR_boxs=False):
    # // TODO should change it to use prepare_region() function.
    sp = doc.get_current_page()

    if OCR_boxs:
        with open(f'{sp.pdf_saving_path}/OCRbox.json') as jf:
            OCR_bbx = json.load(jf, object_hook=keystoint)
        OCR_bbx = [[k[0][0]*sp.pw,k[0][1]*sp.ph,k[0][2]*sp.pw,k[0][3]*sp.ph] for v, k in OCR_bbx.items()]

    if region:
        x = [120, 135]
        y = [50, 75]
        selected_nodes = return_nodes_by_region(sp.nodes_LUT, x, y)
        print(f'found {len(selected_nodes)}')

        canvas = sp.e_canvas  # [x[0]:x[1], y[0]: y[1]]
    else:
        selected_nodes = sp.nodes_LUT.copy()
        canvas = sp.e_canvas

    if paths:
        fig, ax = plt.subplots()
        plt.imshow(canvas)

        for k, v in sp.paths_lst.items():
            path = v.copy()
            if path['p1'] in selected_nodes or path['p2'] in selected_nodes:
                path['p1'] = sp.nodes_LUT[path['p1']]
                path['p2'] = sp.nodes_LUT[path['p2']]
                plot_items([path], coloring='standard')

    if connected_com:
        fig, ax = plt.subplots()
        plt.imshow(sp.e_canvas)
        for k_prime, v_prime in sp.primitives.items():
            paths = return_paths_given_nodes(v_prime, sp.paths_lst, sp.nodes_LUT, replace_nID=True)
            plot_items(paths, coloring='group')

        if OCR_boxs:
            for bb in OCR_bbx:
                rect = patches.Rectangle((bb[0], bb[1]), width=bb[2] - bb[0], height=bb[3] - bb[1], linewidth=1, edgecolor='r',
                                         facecolor='none')
                ax.add_patch(rect)

    # currentAxis = plt.gca()
    # for k_grouped, v_grouped in sp.grouped_prims.items():
    #     x, y, width, height = v_grouped['bbx']
    #     rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='white', facecolor='none',
    #                              linestyle='dashed')
    #     currentAxis.add_patch(rect)


    plt.show()


def plot_txtblocks_regions():
    sp = doc.get_current_page()

    fig, ax = plt.subplots()
    ax.imshow(sp.e_canvas)

    paths_lst = {k: v for k, v in sp.paths_lst.items() if k in sp.nodes_LUT}

    for k, v in paths_lst.items():
        path = v.copy()
        path['p1'] = sp.nodes_LUT[path['p1']]
        path['p2'] = sp.nodes_LUT[path['p2']]
        plot_items([path], coloring='standard')

    # for k, tbl in sp.words_lst.items():
    #     for tb in tbl:
    #         print(tb)
    #         lam = 1.2
    #         w = (tb[2]-tb[0]) + 2
    #         h = (tb[3]-tb[1]) + 2
    #
    #         rect = patches.Rectangle((tb[0]-1, tb[1]-1), width=w, height=h, linewidth=1, edgecolor='r', facecolor='none')
    #         ax.add_patch(rect)

    for k, tb in sp.blocks_lst.items():
        print(tb)
        tb = tb[0]
        lam = 1.2
        w = (tb[2] - tb[0]) + 2
        h = (tb[3] - tb[1]) + 2

        rect = patches.Rectangle((tb[0] - 1, tb[1] - 1), width=w, height=h, linewidth=1, edgecolor='r',
                                 facecolor='none')
        ax.add_patch(rect)


    # plt.show()


def plot_grouped_primes(LC=False, LC_input=False, LC_con=False, Con=False, bbx=False):
    sp = doc.get_current_page()

    fig, ax = plt.subplots()
    ax.imshow(sp.e_canvas)

    if LC:
        selected_prims = {k: v for k, v in sp.grouped_prims.items() if v['cls'] == 'LC'}
        for k_prime, v_prime in selected_prims.items():
            paths = return_paths_given_nodes(v_prime['nodes'], sp.paths_lst, sp.nodes_LUT, replace_nID=True)
            plot_items(paths, coloring='group')

    if LC_input:
        selected_prims = {k: v for k, v in sp.grouped_prims.items() if v['cls'] == 'LC_input'}
        for k_prime, v_prime in selected_prims.items():
            paths = return_paths_given_nodes(v_prime['nodes'], sp.paths_lst, sp.nodes_LUT, replace_nID=True)
            plot_items(paths, coloring='group')

    plt.show()


def get_colors(i):

    colors = ['aliceblue', 'beige', 'bisque', 'blanchedalmond', 'blue', 'blueviolet',
          'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue',
          'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray',
          'darkgreen', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid',
          'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray', 'darkslategrey',
          'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue', 'dimgrey', 'dodgerblue',
          'firebrick', 'forestgreen', 'fuchsia', 'gainsboro', 'gold', 'goldenrod', 'grey', 'green',
          'greenyellow', 'hotpink', 'indianred', 'indigo', 'khaki', 'lawngreen', 'lemonchiffon', 'lightcoral',
          'lightgoldenrodyellow', 'lightgray', 'lightgrey', 'lightgreen', 'lightpink', 'lightsalmon',
          'lightseagreen', 'lightskyblue', 'lightslategray', 'lightslategrey', 'lightsteelblue',
          'lightyellow', 'lime', 'limegreen', 'linen', 'magenta', 'maroon', 'mediumaquamarine', 'mediumblue',
          'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen',
          'mediumturquoise', 'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose', 'moccasin',
          'navy', 'oldlace', 'olive', 'olivedrab', 'orange', 'orangered', 'orchid', 'palegoldenrod',
          'palegreen', 'paleturquoise', 'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink',
          'plum', 'powderblue', 'purple', 'red', 'rosybrown', 'royalblue', 'rebeccapurple', 'saddlebrown',
          'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna', 'silver', 'skyblue', 'slateblue',
          'slategray', 'slategrey', 'snow', 'springgreen', 'steelblue', 'tan', 'teal', 'thistle', 'tomato',
          'turquoise', 'violet', 'wheat', 'yellow', 'yellowgreen']

    return colors[i]

