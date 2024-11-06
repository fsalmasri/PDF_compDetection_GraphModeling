
import os

def get_classes(classes_file):
    classes = []
    with open(classes_file, 'r') as file:
        for line in file:
            classes.append(line.strip())

    return classes

def get_labels_from_file(labels_dir, lbl_name):
    data = []
    try:
        with open(os.path.join(labels_dir, lbl_name), 'r') as file:
            for line in file:
                # Split the line by spaces and convert each part to a float
                numbers = list(map(float, line.split()))
                # Append the list of numbers to the data list
                data.append(numbers)
        return data
    except:
        return None