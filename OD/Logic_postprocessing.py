import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

from LOGIC_POST_utils import get_Polygon, find_LC_coords

from PLCopen import create_main_structure, create_types_block, add_fbd_elements
from lxml import etree


img_path = r'C:\Users\fsalm\Desktop\DISTILL\LOGIC_157_YOLO_v2\test\images\1001_EFF01_10HHG43AA101A_p_0.png'
im = np.array(Image.open(img_path))

predicted_LC_bbxs = np.load('LC_bbx.npy')
predicted_LCCON_bbxs = np.load('LCCON_bbx.npy')



fig, ax = plt.subplots()
plt.imshow(im)

project = create_main_structure()
pous, fbd = create_types_block()
project.append(pous)

import shapely.plotting


# Fill predicted_LCCON_bbxs in Poly_dict
con_dict = {}
for con_ix, con in enumerate(predicted_LCCON_bbxs):
    poly = get_Polygon(con)
    con_dict[con_ix] = poly



for bbx_id, bbx in enumerate(predicted_LC_bbxs):

    x0, y0, xn, yn = find_LC_coords(bbx, im)
    poly_bbx = get_Polygon([x0, y0, xn, yn])
    # shapely.plotting.plot_polygon(poly_bbx)

    LC_selected_cons = {}
    for con_k, con_v in con_dict.items():
        if con_v.intersects(poly_bbx):
            LC_selected_cons[con_k]= list(con_v.centroid.coords)

    add_fbd_elements(fbd, [x0, y0, xn-x0, yn-y0], bbx_id, LC_selected_cons)


xml_content = etree.tostring(project, pretty_print=True, xml_declaration=True, encoding="UTF-8")

with open("plcopen_fbd.xml", "wb") as f:
    f.write(xml_content)
exit()

    # rect2 = patches.Rectangle((x0, y0), xn - x0, yn - y0,
    #                           linewidth=1, edgecolor='r', facecolor='none')
    # ax.add_patch(rect2)

# plt.show()