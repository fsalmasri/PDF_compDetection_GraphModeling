import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random

from . import doc


color_mapping = {'s': 'white', 'f': 'yellow', 'fs': 'blue',
                 'qu': 'purple', 're': 'red', 'c': 'orange', 'test': 'yellow'}

def plot_items(items, coloring = 'standard'):

    path_color = get_colors(random.randint(0, 129))
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


def get_colors(i):

    colors = ['aliceblue', 'beige', 'bisque', 'black', 'blanchedalmond', 'blue', 'blueviolet',
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

