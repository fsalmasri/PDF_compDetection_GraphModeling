import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from . import doc


color_mapping = {'s': 'white', 'f': 'yellow', 'fs': 'blue',
                 'qu': 'purple', 're': 'red', 'c': 'orange', 'test': 'green'}

def plot_items(items):
    for path in items:
        path_color = color_mapping[path['path_type']] if path['item_type'] == 'l' else color_mapping[path['item_type']]
        plt.plot([path['p1'][0], path['p2'][0]], [path['p1'][1], path['p2'][1]], c= path_color)


def plot_graph_nx():
    # print(pos)


    sp = doc.get_current_page()

    pos = {}
    for n in sp.G.nodes:
        pos[n] = np.array(sp.nodes_LUT[n])

    fig, ax = plt.subplots()
    nx.draw(sp.G, pos, with_labels=True, node_size=200, node_color='lightblue',
            font_size=7, font_color='black', font_weight='bold', width=2, ax=ax)

    plt.gca().invert_yaxis()
    plt.axis('on')
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    plt.show()