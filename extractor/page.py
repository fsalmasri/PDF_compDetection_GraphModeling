import numpy as np
from collections import defaultdict
import networkx as nx

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

from . import utils

from . import Saving_path


class page():
    def __init__(self, p, i):
        self.single_page = p
        self.fname = i

        path_to_save = f'{Saving_path}/{self.fname}'
        Path(f"{path_to_save}").mkdir(parents=True, exist_ok=True)

        self.pw = self.single_page.rect.width
        self.ph = self.single_page.rect.height

        # self.line_nodes = []
        self.page_info = {}
        self.nodes_LUT = {}  # a dictionary of nodes list and their coords.
        self.paths_lst = {}  # a dictionary of paths of formate [node 0 , coords, node 1, coords]
        self.words_lst = {}  # a dictionary of text boxes arrange with box id (x0, y0, x1, y1, "word", block_no, line_no, word_no)
        self.blocks_lst = {}  # a dictionary of text boxes arrange with box id (x0, y0, x1, y1, "word", block_no, line_no, word_no)
        self.G = nx.Graph()  # networkx graph network.
        self.primitives = {}
        self.filled_stroke = defaultdict(list)
        self.connected_components = []
        self.grouped_prims ={}


        self.generate_empty_canvas()

    def extract_text(self):
        '''
        Extract text by blocks and words formate and save the two list.
        '''
        for block in self.single_page.get_text('blocks', sort=False):
            if block[5] not in self.blocks_lst:
                self.blocks_lst[block[5]] = []
            self.blocks_lst[block[5]].append(block)

        for word in self.single_page.get_text('words', sort=False):
            if word[5] not in self.words_lst:
                self.words_lst[word[5]] = []
            self.words_lst[word[5]].append(word)


    def generate_empty_canvas(self):
        self.e_canvas = np.zeros((np.ceil(self.ph).astype(int), np.ceil(self.pw).astype(int)))
        self.cv_canvas = np.zeros((np.ceil(self.ph).astype(int), np.ceil(self.pw).astype(int)))


    def store_structued_data(self, item_paths: list):
        '''
        Build a nodes and paths lookup table contains node id and its x, y coords.
        Build Graph networks giving the node ids and the edges.

        :param item_paths: contain a list of paths params in dictionary formate
        {'p1': [x, y], 'p2': [x, y], 'item_type': item_type, 'path_type': path_type}

        :return:
        '''

        for path in item_paths:
            eps = [path['p1'], path['p2']]
            path_nodes = []
            for ep in eps:
                # if nodes exist in the lookup table bring its id given its coordinates.
                if ep in self.nodes_LUT.values():
                    node_id = list(self.nodes_LUT.keys())[list(self.nodes_LUT.values()).index(ep)]
                else:
                    node_id = utils.get_key_id(self.nodes_LUT)

                    self.nodes_LUT[node_id] = ep

                path_nodes.append(node_id)
                # subgraph_nodes.append(node_id)
                self.G.add_node(node_id)
            self.G.add_edge(path_nodes[0], path_nodes[1])

            self.add_to_pathsL(path_nodes, path['item_type'], path['path_type'])

    def store_filled_paths(self, item_paths: list):
        path_id = utils.get_key_id(self.filled_stroke)
        for path in item_paths:
            self.filled_stroke[path_id].append(path)


    def add_to_pathsL(self, nids, item_type, dwg_type):
        path_id = utils.get_key_id(self.paths_lst)
        self.paths_lst[path_id] = {'p1': nids[0], 'p2': nids[1],
                                   'item_type': item_type, 'path_type': dwg_type, 'p_id': 0}

    def extract_paths(self):

        drawings = self.single_page.get_drawings()

        # plt.figure()
        # plt.imshow(self.e_canvas)

        print(f'found {len(drawings)} paths')
        for dwg_idx, dwg in enumerate(drawings[:]):
            dwg_items = dwg['items']
            dwg_type = dwg['type']
            dwg_rect = dwg['rect']
            flag = False

            item_paths = []
            def add_to_main(p1, p2, item_t):
                item_paths.append({'p1': [p1[0], p1[1]], 'p2': [p2[0], p2[1]],
                                   'item_type': item_t, 'path_type': dwg_type})

            for idx, item in enumerate(dwg_items):
                if item[0] == 'l':
                    add_to_main([item[1].x, item[1].y], [item[2].x, item[2].y], item[0])

                if item[0] == 'qu':
                    quads = item[1]

                    add_to_main([quads.ul.x, quads.ul.y], [quads.ur.x, quads.ur.y], item[0])
                    add_to_main([quads.ur.x, quads.ur.y], [quads.lr.x, quads.lr.y], item[0])
                    add_to_main([quads.lr.x, quads.lr.y], [quads.ll.x, quads.ll.y], item[0])
                    add_to_main([quads.ll.x, quads.ll.y], [quads.ul.x, quads.ul.y], item[0])

                if item[0] == 're':
                    rect = item[1]
                    add_to_main([rect.tl.x, rect.tl.y], [rect.tr.x, rect.tr.y], item[0])
                    add_to_main([rect.tr.x, rect.tr.y], [rect.br.x, rect.br.y], item[0])
                    add_to_main([rect.br.x, rect.br.y], [rect.bl.x, rect.bl.y], item[0])
                    add_to_main([rect.bl.x, rect.bl.y], [rect.tl.x, rect.tl.y], item[0])

                if item[0] == 'c':
                    x_coords = [item[1].x, item[2].x, item[3].x, item[4].x]
                    y_coords = [item[1].y, item[2].y, item[3].y, item[4].y]

                    curve_points = utils.get_bezier_cPoints([x_coords, y_coords], num_points=10)
                    for i in range(curve_points.shape[0] - 1): add_to_main(curve_points[i], curve_points[i + 1],
                                                                           item[0])

            # if len(item_paths) > 0 and dwg_type == 'f' and item_paths[-1]['item_type'] == 'l':
            #     item_paths.append({'p1': [item_paths[-1]['p2'][0], item_paths[-1]['p2'][1]],
            #                        'p2': [item_paths[0]['p1'][0], item_paths[0]['p1'][1]],
            #                        'item_type': 'l', 'path_type': dwg_type})
            #
            #     flag = True
            # elif len(item_paths) > 0 and item_paths[-1]['item_type']  == 'c':
            #     flag = True

            # plotter.plot_items(item_paths)
            from .plotter import plot_items
            # plot_items(item_paths)

            if len(item_paths) > 0:
                if flag:
                    self.store_filled_paths(item_paths)
                else:
                    self.store_structued_data(item_paths)

        self.update_primitives_tables()

        # plt.show()

    def build_connected_components(self):
        return list(nx.connected_components(self.G))


    def update_primitives_tables(self, connected_components= None):
        # print(len(self.connected_components))

        # if not self.connected_components:
        #     connected_components = self.build_connected_components()

        if connected_components is None:
            connected_components = self.build_connected_components()


        subgraphs_nodes_lst = [x for x in connected_components if len(x) > 1]
        for subgraph in subgraphs_nodes_lst:

            # add to primitives list
            p_id = utils.get_key_id(self.primitives)
            self.primitives[p_id] = list(subgraph)

            # get all paths ids that contain the nodes in the subgraph
            paths_idx = utils.return_pathsIDX_given_nodes(list(subgraph), self.paths_lst)

            # update the paths LUT with the p_id
            for p in paths_idx:
                self.paths_lst[p]['p_id'] = p_id

    def extract_page_info(self):
        self.page_info = {'width': self.pw, 'height': self.ph}

    def save_images(self):
        pixmap = self.single_page.get_pixmap()
        self.im = utils.pixmap_to_image(pixmap)
        self.svg = self.single_page.get_svg_image()

        self.im.save(f'{Saving_path}/{self.fname}/{self.fname}.png', quality=100, compression=0)
        with open(f'{Saving_path}/{self.fname}/{self.fname}.svg', 'w', encoding='utf-8') as svg_file:
            svg_file.write(self.svg)


    # def study_connected_components(self):
    #
    #     ccomp_byLength = {}
    #     for idx, con_x in enumerate(self.connected_components):
    #         if len(con_x) not in ccomp_byLength:
    #             ccomp_byLength[len(con_x)] = [idx]
    #         else:
    #             ccomp_byLength[len(con_x)].append(idx)
    #
    #     ccomp_byLength = dict(sorted(ccomp_byLength.items()))
    #
    #     s = [[k, len(v)] for k, v in ccomp_byLength.items()]
    #
    #     print(s)
    #     # exit()
    #
    #     for k in ccomp_byLength[28]:
    #         subFig_idx = self.connected_components[k]
    #
    #         H = self.G.subgraph(subFig_idx)
    #         self.plot_graph_nx(H)
    #
    #         paths_idx = utils.return_pathsIDX_given_nodes(subFig_idx, self.paths_lst)
    #         self.plot_paths_by_pathsIDs(paths_idx)
    #
    #
    #         plt.show()
    #
    #     # from collections import Counter
    #     #
    #     # print(Counter(g_len).keys())
    #     # print(Counter(g_len).values())
    #
    #     # print(np.unique(g_len))
    #
    # def find_connectedComp_inRegion(self, x, y):
    #
    #     idx = utils.find_index_by_valueRange(self.nodes_LUT, rng=[x, y])
    #
    #     H = self.G.subgraph(idx)
    #     self.plot_graph_nx(H)
    #
    #     paths_idx = utils.return_pathsIDX_given_nodes(idx, self.paths_lst)
    #     self.plot_paths_by_pathsIDs(paths_idx)
    #
    #     plt.show()
    #
    #
    # def clean_data(self):
    #
    #     nidx = utils.look_in_txtBlocks_dict(self.nodes_LUT, self.words_lst)
    #
    #     paths_idx = utils.return_pathsIDX_given_nodes(nidx, self.paths_lst)
    #     self.plot_paths_by_pathsIDs(paths_idx)
    #
    #     plt.show()
    #
    #
    # General plotters

    # def plot_canvas(self):
    #     plt.figure()
    #     plt.imshow(self.cv_canvas)
    #
    # def plot_two_full_figures(self):
    #     self.plot_graph_nx(self.G)
    #     self.plot_canvas()
    #     plt.show()
    #
    #
    # def plot_paths_by_pathsIDs(self, paths_idx):
    #     paths_lst = {k: v for k, v in self.paths_lst.items() if k in paths_idx}
    #
    #     plt.figure()
    #     for k, v in paths_lst.items():
    #         _, p1, _, p2 = v
    #
    #         plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color='black')
    #
    #     plt.gca().invert_yaxis()
    #
    #
    #
    # def plot_connected_components(self):
    #     # /TODO check function name and behavior.
    #
    #     connected_components = list(nx.connected_components(self.G))
    #
    #     fig = plt.subplots()
    #     plt.imshow(self.e_canvas)
    #
    #     cmap = plt.get_cmap('tab20b')
    #     colors = [cmap(i) for i in np.linspace(0, 1, len(connected_components))]
    #
    #     for i, component in enumerate(connected_components):
    #         # print(f"Connected Component {i + 1}:")
    #
    #         if len(component) > 1:
    #
    #             # fig = plt.subplots()
    #             # plt.imshow(self.e_canvas)
    #
    #             component = list(component)
    #             edges_lst = self.G.edges(component)
    #
    #             x, y = [], []
    #             for e in edges_lst:
    #                 p1, p2 = self.nodes_LUT[e[0]], self.nodes_LUT[e[1]]
    #                 # print(e, p1, p2)
    #                 x.extend([p1[0], p2[0]])
    #                 y.extend([p1[1], p2[1]])
    #                 plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color=colors[i]) #c='white'
    #
    #             # plt.plot(x, y)
    #             # plt.show()
    #
    #     #     for line in lines:
    #     #         if any(endpoint in component for endpoint in line):
    #     #             print(line)
    #
    #     plt.show()
    #
    # def plot_txtblocks_regions(self):
    #     fig, ax = plt.subplots()
    #     ax.imshow(self.e_canvas)
    #
    #     paths_lst = {k: v for k, v in self.paths_lst.items() if k in self.nodes_LUT}
    #
    #     for k, v in paths_lst.items():
    #         _, p1, _, p2 = v
    #
    #         ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='white')
    #
    #     # plt.gca().invert_yaxis()
    #
    #     for k, tbl in self.words_lst.items():
    #         for tb in tbl:
    #             print(tb)
    #             lam = 1.2
    #             w = (tb[2]-tb[0]) + 2
    #             h = (tb[3]-tb[1]) + 2
    #
    #             rect = patches.Rectangle((tb[0]-1, tb[1]-1), width=w, height=h, linewidth=1, edgecolor='r', facecolor='none')
    #             ax.add_patch(rect)
    #
    #     plt.show()


    # General Utils.
    def save_data(self):
        utils.save_data(f'{Saving_path}/{self.fname}',
                        self.G, self.nodes_LUT, self.paths_lst, self.words_lst, self.blocks_lst,
                        self.primitives, dict(self.filled_stroke), self.grouped_prims, self.page_info)

    def load_data(self):

        (self.G, self.nodes_LUT, self.paths_lst, self.words_lst, self.blocks_lst,
         self.primitives, self.filled_stroke, self.grouped_prims, self.page_info) = (
            utils.load_data(f'{Saving_path}/{self.fname}'))

        self.build_connected_components()



