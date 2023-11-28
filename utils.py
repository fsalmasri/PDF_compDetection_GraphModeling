import numpy

def find_index_by_valueRange(LUT, rng: list):

    x, y = rng
    idx = []
    for k, v in LUT.items():
        if v[0] >= x[0] and v[0] <= x[1]:
            if v[1] >= y[0] and v[1] <= y[1]:
                idx.append(k)

    return idx


def return_values_by_Idx(LUT, nodes):
    pos = {}
    for n in nodes:
        pos[n] = LUT[n]

    return pos


def keystoint(x):
    return {int(k): v for k, v in x.items()}
