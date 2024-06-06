import numpy as np
import trimesh


def generate_transform_matrix(steps, axis):

    matrix = trimesh.transformations.rotation_matrix(2*np.pi/steps, axis)
    return matrix


def gen_object_tag(axis_angle, threshold):

    tag = {"symmetries_discrete": [],
           "symmetries_continuous": []}

    # x axis:
    axis = np.array([1, 0, 0])
    if axis_angle[0] >= threshold:
        con = {"axis": [1, 0, 0], "offset": [0, 0, 0]}
        tag["symmetries_continuous"].append(con)
    elif axis_angle[0] > 1:
        h = generate_transform_matrix(axis_angle[0], axis).flatten()
        tag["symmetries_discrete"].append(h.tolist())

    # x axis:
    axis = np.array([0, 1, 0])
    if axis_angle[1] >= threshold:
        con = {"axis": [0, 1, 0], "offset": [0, 0, 0]}
        tag["symmetries_continuous"].append(con)
    elif axis_angle[1] > 1:
        h = generate_transform_matrix(axis_angle[1], axis).flatten()
        tag["symmetries_discrete"].append(h.tolist())

    # x axis:
    axis = np.array([0, 0, 1])
    if axis_angle[2] >= threshold:
        con = {"axis": [0, 0, 1], "offset": [0, 0, 0]}
        tag["symmetries_continuous"].append(con)
    elif axis_angle[2] > 1:
        h = generate_transform_matrix(axis_angle[2], axis).flatten()
        tag["symmetries_discrete"].append(h.tolist())

    # Clean dictionary
    if len(tag["symmetries_discrete"]) == 0:
        del tag["symmetries_discrete"]
    if len(tag["symmetries_continuous"]) == 0:
        del tag["symmetries_continuous"]
    return tag

