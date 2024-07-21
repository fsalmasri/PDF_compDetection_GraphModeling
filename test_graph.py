import pickle
import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

flst = os.listdir('graph_set')
conf_mx = np.zeros((len(flst), len(flst)))

all = []
for f in flst:
    all.append(pickle.load(open(f'graph_set/{f}', 'rb')))

for idx in range(len(flst)-1):
    G1 = all[idx]
    # G1 = pickle.load(open(f'graph_set/{flst[idx]}', 'rb'))
    pos = nx.get_node_attributes(G1, 'pos')
    fig = plt.figure()
    nx.draw(G1, pos, with_labels=True)
    plt.savefig(f'graph_imgs/{flst[idx][:-6]}png')
    plt.close(fig)
    # for jdx in range(idx +1, len(flst)):
    #     G1 = all[idx]
    #     G2 = all[jdx]
    #
    #     # print(idx, jdx, G1, G2)
    #     # G2 = pickle.load(open(f'graph_set/{flst[jdx]}', 'rb'))
    #     dist = nx.optimize_graph_edit_distance(G1, G2)
    #     dist = min(list(dist))
    #
    #     conf_mx[idx, jdx] = dist
    #     conf_mx[jdx, idx] = dist
    #
    #     print(idx, jdx, dist)



# plt.imshow(conf_mx, cmap='')
# plt.show()
    # print(G)
    # print(pos)

    # plt.figure()
    # nx.draw(G, pos, with_labels=True)
    # plt.show()
    # # exit()

exit()