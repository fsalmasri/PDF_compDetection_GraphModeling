
import numpy as np
import fitz
import networkx as nx
import json

import matplotlib.pyplot as plt
import cv2
import utils


class page():
    def __init__(self, p):
        self.single_page = p

        self.pw = self.single_page.rect.width
        self.ph = self.single_page.rect.height

        # self.line_nodes = []
        self.lookupTable = {}
        self.paths_lst = {}

        self.G = nx.Graph()

        self.generate_empty_canvas()

    def extract_text(self):
        # //TODO later
        text = self.single_page.get_text(delimiters='\n')  # .encode("utf8")
        print(type(text))
        print(text)
        # for t in text:
        #     print(t)

    def generate_empty_canvas(self):
        # print(np.ceil(self.pw).astype(int), np.ceil(self.ph).astype(int))

        self.e_canvas = np.zeros((np.ceil(self.ph).astype(int), np.ceil(self.pw).astype(int)))
        self.cv_canvas = np.zeros((np.ceil(self.ph).astype(int), np.ceil(self.pw).astype(int)))



    def store_structued_data(self, eps: list):
        nodes = []
        for ep in eps:
            if ep in self.lookupTable.values():
                node_id = list(self.lookupTable.keys())[list(self.lookupTable.values()).index(ep)]
            else:
                if not self.lookupTable.keys():
                    node_id = 1
                else:
                    node_id = list(self.lookupTable.keys())[-1]+1
                self.lookupTable[node_id] = ep

            nodes.append(node_id)

            # assuming only path. then it only has two nodes.
            self.G.add_node(node_id)

        self.add_to_pathsL(nodes, eps)
        self.G.add_edge(nodes[0], nodes[1])


    def add_to_pathsL(self, nids, eps):

        # //TODO move this to function
        if not self.paths_lst.keys():
            path_id = 1
        else:
            path_id = list(self.paths_lst.keys())[-1] + 1

        self.paths_lst[path_id] = [nids[0], eps[0], nids[1], eps[1]]

    def extract_paths(self, fname):

        # plt.figure()
        # plt.imshow(self.e_canvas)

        drawings = self.single_page.get_drawings()
        print(f'found {len(drawings)} pathes')
        # exit()
        for d in drawings[:2000]:
            for idx, ld in enumerate(d['items']):
                # print(ld)

                if 'l' in ld:
                    ep1 = (ld[1].x, ld[1].y)
                    ep2 = (ld[2].x, ld[2].y)

                    # plt.plot([ld[1].x, ld[2].x], [ld[1].y, ld[2].y], c='white')

                    p1 = (int(ld[1].x), int(ld[1].y))
                    p2 = (int(ld[2].x), int(ld[2].y))

                    cv2.line(self.cv_canvas, p1, p2, color=255, thickness=1)
                    self.store_structued_data([list(ep1), list(ep2)])


        self.build_connected_components()
        self.save_data(fname)


    def save_data(self, fname):
        with open(f"data/LUT_{fname}.json", "w") as jf:
            json.dump(self.lookupTable, jf)

        with open(f"data/PathsL_{fname}.json", "w") as jf:
            json.dump(self.paths_lst, jf)

        nx.write_graphml_lxml(self.G, f"data/graph_{fname}.graphml")

    def load_data(self, fname):
        self.G = nx.read_graphml(f"data/graph_{fname}.graphml")
        nodes_list = [int(x) for x in self.G.nodes]
        self.G.nodes = nodes_list
        edges_list = [(int(x), int(y)) for x, y in self.G.edges]

        self.G.remove_edges_from(list(self.G.edges()))
        self.G.add_edges_from(edges_list)

        self.build_connected_components()


        with open('data/LUT.json') as jf:
            self.lookupTable = json.load(jf, object_hook=utils.keystoint)

        with open('data/PathsL.json') as jf:
            self.paths_lst = json.load(jf, object_hook=utils.keystoint)


    def plot_canvas(self):
        plt.figure()
        plt.imshow(self.cv_canvas)

    def plot_graph_nx(self, G):
        # pos = nx.spring_layout(self.G)
        # print(pos)

        pos = {}
        for n in G.nodes:
            pos[n] = np.array(self.lookupTable[n])

        fig, ax = plt.subplots()
        nx.draw(G, pos, with_labels=True, node_size=200, node_color='lightblue',
                font_size=7, font_color='black', font_weight='bold', width=2, ax=ax)

        plt.gca().invert_yaxis()
        plt.axis('on')
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        # plt.show()

    def plot_paths_by_pathsIDs(self, paths_idx):
        paths_lst = {k: v for k, v in self.paths_lst.items() if k in paths_idx}

        plt.figure()
        for k, v in paths_lst.items():
            _, p1, _, p2 = v

            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color='black')

        plt.gca().invert_yaxis()


    def plot_two_full_figures(self):
        self.plot_graph_nx(self.G)
        self.plot_canvas()
        plt.show()


    def plot_connected_components(self):
        # /TODO check function name and behavior.

        connected_components = list(nx.connected_components(self.G))

        fig = plt.subplots()
        plt.imshow(self.e_canvas)

        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i) for i in np.linspace(0, 1, len(connected_components))]

        for i, component in enumerate(connected_components):
            # print(f"Connected Component {i + 1}:")

            if len(component) > 1:

                # fig = plt.subplots()
                # plt.imshow(self.e_canvas)

                component = list(component)
                edges_lst = self.G.edges(component)

                x, y = [], []
                for e in edges_lst:
                    p1, p2 = self.lookupTable[e[0]], self.lookupTable[e[1]]
                    # print(e, p1, p2)
                    x.extend([p1[0], p2[0]])
                    y.extend([p1[1], p2[1]])
                    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color=colors[i]) #c='white'

                # plt.plot(x, y)
                # plt.show()

        #     for line in lines:
        #         if any(endpoint in component for endpoint in line):
        #             print(line)

        plt.show()


    def build_connected_components(self):
        self.connected_components = list(nx.connected_components(self.G))


    def study_connected_components(self):

        ccomp_byLength = {}
        for idx, con_x in enumerate(self.connected_components):
            if len(con_x) not in ccomp_byLength:
                ccomp_byLength[len(con_x)] = [idx]
            else:
                ccomp_byLength[len(con_x)].append(idx)

        ccomp_byLength = dict(sorted(ccomp_byLength.items()))

        s = [[k, len(v)] for k, v in ccomp_byLength.items()]

        print(s)
        # exit()

        for k in ccomp_byLength[28]:
            subFig_idx = self.connected_components[k]

            H = self.G.subgraph(subFig_idx)
            self.plot_graph_nx(H)

            paths_idx = utils.return_path_given_nodes(subFig_idx, self.paths_lst)
            self.plot_paths_by_pathsIDs(paths_idx)


            plt.show()

        # from collections import Counter
        #
        # print(Counter(g_len).keys())
        # print(Counter(g_len).values())

        # print(np.unique(g_len))

    def find_connectedComp_inRegion(self, x, y):

        idx = utils.find_index_by_valueRange(self.lookupTable, rng=[x, y])

        H = self.G.subgraph(idx)
        self.plot_graph_nx(H)

        paths_idx = utils.return_path_given_nodes(idx, self.paths_lst)
        self.plot_paths_by_pathsIDs(paths_idx)

        plt.show()



