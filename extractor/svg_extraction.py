import xml.etree.ElementTree as ET
from svg.path import parse_path
from svg.path.path import Line, CubicBezier
import svgwrite
from xml.dom import minidom
import numpy as np

import matplotlib.pyplot as plt

from . import Saving_path, LS_path

region_x = 40
region_y = 40
region_width = 2837
region_height = 1965
sub_region_x = 1550
sub_region_y = 1660

def get_regions(svg_root):

    width = svg_root.getAttribute('width')
    height = svg_root.getAttribute('height')

    if width.endswith('pt'):
        width = float(width[:-2])
    if height.endswith('pt'):
        height = float(height[:-2])

    if width == 2837 and height == 1965:
        region_x = 40
        region_y = 40
        region_width = 2837
        region_height = 1965
        sub_region_x = 1550
        sub_region_y = 1660

    else:
        raise NotImplementedError

    return [region_x, region_y, region_width, region_height, sub_region_x, sub_region_y]


im = np.zeros((1965, 2837))
def transform_points(x, y, transform_matrix_data):

    transform_matrix = list(map(float, transform_matrix_data[7:-1].split(',')))

    transform_matrix = np.array([[transform_matrix[0], transform_matrix[2], transform_matrix[4]],
                                 [transform_matrix[1], transform_matrix[3], transform_matrix[5]],
                                 [0, 0, 1]])

    point_vector = np.array([x, y, 1])
    transformed_point = np.dot(transform_matrix, point_vector)
    transformed_x, transformed_y, _ = transformed_point

    return transformed_x, transformed_y
#
#             def check_PointRange_2(x, y, rng=[[140, 151], [490, 510]]): #[100, 170], [460, 560]
#                 return x > rng[0][0] and x < rng[0][1] and y > rng[1][0] and y < rng[1][1]

def is_point_inside_region(node, regions):

    region_x, region_y, region_width, region_height, sub_region_x, sub_region_y = regions

    flag = True
    if (
            region_x <= node[0] <= region_x + region_width and
            region_y <= node[1] <= region_y + region_height
    ):
        flag =  False
    if(
            sub_region_x <= node[0] and sub_region_y <= node[1]
    ):
        flag = True

    return flag

def parse(path_element, transform_matrix_data = None):

    path_data = path_element.getAttribute('d')
    if not transform_matrix_data:
        transform_matrix_data = path_element.getAttribute('transform')
    parsed_path = parse_path(path_data)
    nodes = []

    if transform_matrix_data:
        for element in parsed_path:
            if isinstance(element, Line):
                x0, y0 = element.start.real, element.start.imag
                x1, y1 = element.end.real, element.end.imag

                x0, y0 = transform_points(x0, y0, transform_matrix_data)
                x1, y1 = transform_points(x1, y1, transform_matrix_data)

                nodes.append([x0, y0])
                nodes.append([x1, y1])

            elif isinstance(element, CubicBezier):
                for point in [element.start, element.control1, element.control2, element.end]:
                    x, y = point.real, point.imag
                    x, y = transform_points(x, y, transform_matrix_data)
                    nodes.append([x, y])

    return nodes

def find_path_by_id(path_elements, path_id):
    for path_element in path_elements:
        if path_element.getAttribute('id') == path_id:
            return path_element
    return None


def clean_borders_svg(fname):

    doc = minidom.parse(f'{Saving_path}/{fname}/{fname}.svg')
    svg_root = doc.getElementsByTagName('svg')[0]

    regions = get_regions(svg_root)

    path_elements = svg_root.getElementsByTagName('path')
    image_elements = svg_root.getElementsByTagName('image')

    use_elements = doc.getElementsByTagName('use')

    for use_element in use_elements:
        data_text = use_element.getAttribute('data-text')
        xlink_href = use_element.getAttribute('xlink:href')
        transform_matrix_data = use_element.getAttribute('transform')
        flag = False

        if data_text and xlink_href.startswith('#font_'):
            selected_path_element = find_path_by_id(path_elements, xlink_href[1:])
            if selected_path_element:
                parsed_nodes = parse(selected_path_element, transform_matrix_data)
                if parsed_nodes:
                    for node in parsed_nodes:
                        if is_point_inside_region(node, regions):
                            flag = True
        if flag:
            parent = use_element.parentNode
            parent.removeChild(use_element)


    if image_elements:
        print("Image elements exist in the SVG.")
        for image_element in image_elements:
            print("Remove Image ID:", image_element.getAttribute('id'))
            parent = image_element.parentNode
            parent.removeChild(image_element)


    for path_element in path_elements:
        parsed_nodes = parse(path_element)
        flag = False
        if parsed_nodes:
            for node in parsed_nodes:
                if is_point_inside_region(node, regions):
                    flag = True

        if flag:
            parent = path_element.parentNode
            parent.removeChild(path_element)

    output_svg_file = f'{LS_path}/images/{fname}_cleaned.svg'
    with open(output_svg_file, 'w') as output_file:
        output_file.write(doc.toprettyxml())



