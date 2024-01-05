
import matplotlib.pyplot as plt

color_mapping = {'s': 'white', 'f': 'yellow', 'fs': 'blue',
                 'qu': 'purple', 're': 'red', 'c': 'orange', 'test': 'green'}

def plot_items(items):
    for path in items:
        path_color = color_mapping[path['path_type']] if path['item_type'] == 'l' else color_mapping[path['item_type']]
        plt.plot([path['p1'][0], path['p2'][0]], [path['p1'][1], path['p2'][1]], c= path_color)