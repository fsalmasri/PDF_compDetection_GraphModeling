
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

    def extract_text(self):
        # //TODO later
        text = self.single_page.get_text(delimiters='\n')  # .encode("utf8")
        print(type(text))
        print(text)
        # for t in text:
        #     print(t)

    def generate_empty_canvas(self):
        print(np.ceil(self.pw).astype(int), np.ceil(self.ph).astype(int))

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

    def extract_paths(self):

        # plt.figure()
        # plt.imshow(self.e_canvas)

        drawings = self.single_page.get_drawings()
        for d in drawings: #[:2000]:
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


        # //TODO save LUTT, Graph, Paths.
        self.save_data()

    def save_data(self):
        with open("data/LUT.json", "w") as fp:
            json.dump(self.lookupTable, fp)

        with open("data/PathsL.json", "w") as fp:
            json.dump(self.paths_lst, fp)

        nx.write_graphml_lxml(self.G, "data/graph.graphml")
        # G = nx.read_graphml


    def plot_canvas(self):
        plt.figure()
        plt.imshow(self.e_canvas)

    def plot_graph_nx(self, G):
        # pos = nx.spring_layout(self.G)
        # print(pos)

        pos = {}
        for n in G.nodes:
            pos[n] = np.array(self.lookupTable[n])

        fig, ax = plt.subplots()
        nx.draw(G, pos, with_labels=True, node_size=200, node_color='lightblue',
                font_size=10, font_color='black', font_weight='bold', width=2, ax=ax)

        plt.gca().invert_yaxis()
        plt.axis('on')
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        # plt.show()


    def plot_two_full_figures(self):
        self.plot_graph_nx(self.G)
        self.plot_canvas()
        plt.show()


    def build_connected_components(self):
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


    def find_connectedComp_inRegion(self, x, y):

        idx = utils.find_index_by_valueRange(self.lookupTable, rng=[x, y])
        H = nx.subgraph(self.G, idx)
        self.plot_graph_nx(H)


        pos = utils.return_values_by_Idx(self.lookupTable, H.nodes)
        pos = np.array(list(pos.values()))
        pos = np.vstack([pos, pos[0]])

        plt.figure()
        plt.plot(pos[:,0], pos[:,1])
        plt.gca().invert_yaxis()

        plt.show()
        exit()


