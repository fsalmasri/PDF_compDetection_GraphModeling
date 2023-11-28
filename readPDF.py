import PyPDF2

import fitz
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rc('image', cmap='gray')

import matplotlib.patches as mpatches
import matplotlib.path as mpath
Path = mpath.Path

from bezier import bezier
import cv2





# file = open(pdfpath, 'rb')





# plt.figure(figsize=(np.ceil(pw//2).astype(int), np.ceil(ph//2).astype(int)), dpi=80)

# fig, ax = plt.subplots()



# print(page.get_drawings())
for d in page.get_drawings()[1000:1010]:

    for idx, ld in enumerate(d['items']):
        if 'l' in ld:
            # p1 = (int(ld[1].x), int(ld[1].y))
            # p2 = (int(ld[2].x), int(ld[2].y))

            # plt.plot([ld[1].x, ld[2].x], [ld[1].y, ld[2].y], c='white')
            # cv2.line(im, p1, p2, color=255, thickness=1)

            ep1 = (ld[1].x, ld[1].y)
            ep2 = (ld[2].x, ld[2].y)
            G.add_node(ep1)
            G.add_node(ep2)
            G.add_edge(ep1, ep2)

        if 'c' in ld:
            # print(ld)
            curve = bezier.Curve([[ld[1].x, ld[2].x, ld[3].x, ld[4].x],
                                 [ld[1].y, ld[2].y, ld[3].y, ld[4].y]], degree=3)

            t_values = np.linspace(0.0, 1.0, 1000)  # Increase the number of points
            nodes = curve.evaluate_multi(t_values)

            # nodes = curve.nodes
            x_points = nodes[0, :]
            y_points = nodes[1, :]

            # plt.plot(x_points, y_points, c='red')
            #
            # pp1 = mpatches.PathPatch(
            #     Path([(ld[1].x, ld[1].y), (ld[2].x, ld[2].y),
            #           (ld[3].x, ld[3].y), (ld[4].x, ld[4].y)],
            #          [Path.MOVETO, Path.CURVE3, Path.CURVE3, Path.CLOSEPOLY]),
            #     fc="none", transform=ax.transData)

            # ax.add_patch(pp1)
# plt.imshow(im)

pos = nx.spring_layout(G)

plt.figure()
# Draw the graph
# nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightblue', font_size=10, font_color='black', font_weight='bold')
# nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color='blue', width=2)

nx.draw(G, pos, with_labels=True, node_size=200, node_color='lightblue',
        font_size=1, font_color='black', font_weight='bold', width=2)


plt.figure()
plt.imshow(im)

connected_components = list(nx.connected_components(G))
for i, component in enumerate(connected_components):
    print(f"Connected Component {i + 1}:")
    print(len(component), component)

    for c in component:
        # plt.plot([c[1].x, ld[2].x], [ld[1].y, ld[2].y], c='white')

        print(c)
        exit()
#     for line in lines:
#         if any(endpoint in component for endpoint in line):
#             print(line)


# plt.show()


