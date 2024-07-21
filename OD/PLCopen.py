from lxml import etree
from datetime import datetime

def create_main_structure():
    """
    Creates the main structure of the PLCopen XML file.

    :return: The root element of the XML tree and the body element where FBD elements will be added.
    """
    # Create the root element
    project = etree.Element("project", xmlns="http://www.plcopen.org/xml/tc6_0201")

    # Get the current datetime
    current_datetime = datetime.now()

    # Add file header with the specified attributes
    etree.SubElement(project, "fileHeader",
                     companyName="Unknown",
                     companyURL="",
                     productName="Unknown",
                     productVersion="1",
                     productRelease="",
                     creationDateTime=current_datetime.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                     contentDescription="")

    # Add content header with the provided attributes
    content_header = etree.SubElement(project, "contentHeader",
                                      name="Unknown",
                                      version="",
                                      organization="",
                                      author="",
                                      language="english")
    coordinate_info = etree.SubElement(content_header, "coordinateInfo")

    # Add scaling information for FBD, LD, and SFC
    for diagram_type in ["fbd", "ld", "sfc"]:
        dtype = etree.SubElement(coordinate_info, diagram_type)
        scaling = etree.SubElement(dtype, "scaling")
        scaling.set("x", "5")
        scaling.set("y", "5")

    # Return the root element
    return project


def create_types_block():
    """
    Creates the types block of the PLCopen XML file.

    :return: The types element containing dataTypes and an empty pous element.
    """
    # Create the types element
    types = etree.Element("types")

    # Add dataTypes element
    etree.SubElement(types, "dataTypes")

    # Add an empty pous element
    pous = etree.SubElement(types, "pous")

    pou = etree.SubElement(pous, "pou")
    pou.set('name', f'pou1')
    pou.set("pouType", "program")
    interface = etree.SubElement(pou, "interface")
    body = etree.SubElement(pou, "body")
    fbd = etree.SubElement(body, "FBD")

    # This must be returned or call another function
    inVariable = etree.SubElement(fbd, "inVariable")
    outVariable = etree.SubElement(fbd, "outVariable")

    return pous, fbd


def add_fbd_elements(parent, bounding_boxe, i, LC_selected_cons):
    """
    Adds FBD elements to the provided parent element.

    :param parent: The parent element to which FBD elements will be added.
    :param bounding_boxes: A list of tuples representing the bounding boxes (x, y, width, height).
    """
    x, y, width, height = bounding_boxe


    block = etree.SubElement(parent, "block", localId=f'{i}', height=str(height), width=str(width), typeName="AND")
    etree.SubElement(block, "position", x=str(x), y=str(y))
    invars = etree.SubElement(block, "inputVariables")

    for idx, con in enumerate(LC_selected_cons):
        con_x = LC_selected_cons[con][0][0]
        con_y = LC_selected_cons[con][0][1]

        var = etree.SubElement(invars, "variable", formalParameter= f"IN{idx}")
        con_point = etree.SubElement(var, "connectionPointIn")
        rel_position = etree.SubElement(con_point, "relPosition")
        rel_position.set("x", f'{con_x - x}')
        rel_position.set("y", f'{con_y - y}')


