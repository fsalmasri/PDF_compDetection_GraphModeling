import xml.etree.ElementTree as ET
from svg.path import parse_path
from svg.path.path import Line, CubicBezier
import svgwrite
from xml.dom import minidom


region_x = 40
region_y = 40
region_width = 1510
region_height = 1620


def is_path_inside_region(path_data):
    path = parse_path(path_data)
    for element in path:
        if isinstance(element, Line):
            x1, y1 = element.start.real, element.start.imag
            x2, y2 = element.end.real, element.end.imag

            print(x1, y1, x2, y2)
            exit()
            if (
                    region_x <= x1 <= region_x + region_width and
                    region_y <= y1 <= region_y + region_height
            ) or (
                    region_x <= x2 <= region_x + region_width and
                    region_y <= y2 <= region_y + region_height
            ):
                return True

        elif isinstance(element, CubicBezier):
            for point in [element.start, element.control1, element.control2, element.end]:
                x, y = point.real, point.imag
                if region_x <= x <= region_x + region_width and region_y <= y <= region_y + region_height:
                    return True
        else:
            return False


def clean_borders_svg():
    # tree = ET.parse('../data/0/0_copy.svg')
    # root = tree.getroot()
    paths_to_remove = []
    #
    # # Find all path elements in the SVG file
    # paths = root.findall('.//{http://www.w3.org/2000/svg}path')
    #
    # for idx, path in enumerate(paths):
    #     path_data = path.get('d')
    #     if not is_path_inside_region(path_data):
    #         paths_to_remove.append(path)
    #         root.remove(path)
    #         print(idx)
    # # exit()
    # for path in paths_to_remove:
    #     root.remove(path)

    # tree.write('output.svg')

    doc = minidom.parse('../data/0/0_copy.svg')
    svg_root = doc.getElementsByTagName('svg')[0]
    path_elements = svg_root.getElementsByTagName('path')
    for path_element in path_elements:
        path_data = path_element.getAttribute('d')
        if not is_path_inside_region(path_data):
            parent = path_element.parentNode
            parent.removeChild(path_element)


    # tree.write('output.svg')
    output_svg_file = 'output.svg'
    with open(output_svg_file, 'w') as output_file:
        output_file.write(doc.toprettyxml())

    with open("output.xml", "w") as fs:
        fs.write(doc.toxml())
        fs.close()

clean_borders_svg()

# def study_paths_svg():
#     sp = doc.get_current_page()
#
#     # from PIL import Image
#     # import cairosvg
#     # from io import BytesIO
#     # import pysvg
#     # import xml.etree.ElementTree as ET
#     from xml.dom import minidom
#     from svg.path import parse_path
#     from svg.path.path import Line, Move, CubicBezier
#
#     svg_image = sp.single_page.get_svg_image()
#     svg_dom = minidom.parseString(svg_image)
#
#
#     clip_paths = svg_dom.getElementsByTagName('clipPath')
#     g_tag = svg_dom.getElementsByTagName('g')
#
#     for g in g_tag:
#         group_id = g.getAttribute('id')
#         path_elements = g.getElementsByTagName('path')
#
#         plt.figure()
#         plt.imshow(sp.e_canvas)
#
#         # Iterate through path elements
#         for path_element in path_elements[:]:
#             path_data = path_element.getAttribute('d')  # Get the 'd' attribute containing the path data
#
#             transform_matrix_data = path_element.getAttribute('transform')
#             transform_matrix = list(map(float, transform_matrix_data[7:-1].split(',')))
#
#             transform_matrix = np.array([[transform_matrix[0], transform_matrix[2], transform_matrix[4]],
#                                          [transform_matrix[1], transform_matrix[3], transform_matrix[5]],
#                                          [0, 0, 1]])
#
#             parsed_path = parse_path(path_data)
#
#             # Print the clipPath ID and path data line by line
#             # print(f"Group ID: {group_id}")
#             # print(f"Path Data: {path_data}")
#             # print(f"Parsed Path Data: {parsed_path}")
#             # print(transform_matrix)
#
#             def transform_points(x, y, transform_matrix):
#                 point_vector = np.array([x, y, 1])
#                 transformed_point = np.dot(transform_matrix, point_vector)
#                 transformed_x, transformed_y, _ = transformed_point
#
#                 return transformed_x, transformed_y
#
#             def check_PointRange_2(x, y, rng=[[140, 151], [490, 510]]): #[100, 170], [460, 560]
#                 return x > rng[0][0] and x < rng[0][1] and y > rng[1][0] and y < rng[1][1]
#
#             flag = False
#             for e in parsed_path:
#                 if isinstance(e, Line):
#                     x0 = e.start.real
#                     y0 = e.start.imag
#                     x1 = e.end.real
#                     y1 = e.end.imag
#
#                     # print(path_data, transform_matrix_data)
#                     # print("(%.2f, %.2f) - (%.2f, %.2f)" % (x0, y0, x1, y1))
#
#                     x0, y0 = transform_points(x0, y0, transform_matrix)
#                     x1, y1 = transform_points(x1, y1, transform_matrix)
#
#                     # print("(%.2f, %.2f) - (%.2f, %.2f)" % (x0, y0, x1, y1))
#
#                     if check_PointRange_2(x0, y0) and check_PointRange_2(x1, y1):
#                         flag = True
#                         # print(e)
#                         print("(%.2f, %.2f) - (%.2f, %.2f)" % (x0, y0, x1, y1))
#                         plt.plot([x0, x1], [y0, y1], c='white')
#
#
#                 elif isinstance(e, Move):
#                     pass
#
#                 elif isinstance(e, CubicBezier):
#                     pass
#
#                 else:
#                     pass
#
#             if flag:
#
#                 import svgpathtools
#
#                 print(path_element.attributes.items())
#                 print(svgpathtools.parse_path(path_data))
#                 exit()
#                 # for e in parsed_path:
#                 #     print(e)
#
#                 print("--------------------")
#
#
#         plt.show()
#         exit()
#
#     exit()