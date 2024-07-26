import json

import matplotlib.pyplot as plt
import networkx as nx
import functools

from . import utils

def save_as_json(filename_attr):
    """
    A decorator to save a class attribute to a JSON file.

    :param filename_attr: The name of the file to save data in (without the .json extension)
    """

    def decorator(method):
        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            data = method(self, *args, **kwargs)
            filepath = f"{self.pdf_saving_path}/{filename_attr}.json"
            with open(filepath, "w") as jf:
                json.dump(data, jf, indent=4)
        return wrapper

    return decorator


def load_from_json(filename_attr):
    """
    A decorator to load data from a JSON file and assign it to a class attribute.

    :param filename_attr: The name of the file to load data from (without the .json extension)
    """

    def decorator(method):
        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            filepath = f"{self.pdf_saving_path}/{filename_attr}.json"
            try:
                with open(filepath, "r") as jf:
                    data = json.load(jf, object_hook=utils.keystoint)
                    method(self, data)
            except FileNotFoundError:
                print(f"File {filepath} not found.")
            except json.JSONDecodeError:
                print(f"Error decoding JSON from file {filepath}.")

        return wrapper

    return decorator

class PageDefaultMixin:

    pdf_saving_path = None

    # Loading functions
    @load_from_json('info')
    def load_info(self, data):
        self.page_info = data

    @load_from_json('grouped_prims')
    def load_grouped_prims(self, data):
        self.grouped_prims = data

    @load_from_json('nodes_LUT')
    def load_nodes_LUT(self, data):
        self.nodes_LUT = data

    @load_from_json('paths_LUT')
    def load_paths_lst(self, data):
        self.paths_lst = data

    @load_from_json('textBox')
    def load_words_lst(self, data):
        self.words_lst = data

    @load_from_json('blockBox')
    def load_blocks_lst(self, data):
        self.blocks_lst = data

    @load_from_json('primitives')
    def load_primitives(self, data):
        self.primitives = data

    @load_from_json('filled_stroke')
    def load_filled_stroke(self, data):
        self.filled_stroke = data

    def load_G(self):
        filepath = f"{self.pdf_saving_path}/graph.graphml"
        try:
            self.G = nx.read_graphml(filepath)
        except FileNotFoundError:
            print(f"File {filepath} not found.")
        except nx.NetworkXError:
            print(f"Error reading graph from file {filepath}.")

    # Saving functions
    @save_as_json('info')
    def save_info(self):
        return self.page_info

    @save_as_json('grouped_prims')
    def save_grouped_prims(self):
        return self.grouped_prims

    @save_as_json('nodes_LUT')
    def save_nodes_LUT(self):
        return self.nodes_LUT

    @save_as_json('paths_LUT')
    def save_paths_lst(self):
        return self.paths_lst

    @save_as_json('textBox')
    def save_words_lst(self):
        return self.words_lst

    @save_as_json('blockBox')
    def save_blocks_lst(self):
        return self.blocks_lst

    @save_as_json('primitives')
    def save_primitives(self):
        return self.primitives

    @save_as_json('filled_stroke')
    def save_filled_stroke(self):
        return self.filled_stroke

    def save_G(self):
        nx.write_graphml_lxml(self.G, f"{self.pdf_saving_path}/graph.graphml")



    def save_images(self, dpi=150):
        pixmap = self.single_page.get_pixmap(dpi=dpi)
        im = utils.pixmap_to_image(pixmap)
        # svg = self.single_page.get_svg_image()
        im.save(f'{self.pdf_saving_path}/img.png', quality=100, compression=0)



    def load_data(self):
        print('Loading all data ...')
        self.load_info()
        self.load_grouped_prims()
        self.load_nodes_LUT()
        self.load_paths_lst()
        self.load_words_lst()
        self.load_blocks_lst()
        self.load_primitives()
        self.load_filled_stroke()
        self.load_G()
        print('Data loaded.')

    def save_data(self):
        print('Saving all Lists ...')
        self.save_info()
        self.save_grouped_prims()
        self.save_nodes_LUT()
        self.save_paths_lst()
        self.save_words_lst()
        self.save_blocks_lst()
        self.save_primitives()
        self.save_filled_stroke()
        self.save_G()
