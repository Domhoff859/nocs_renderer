import trimesh
import numpy as np
import scipy as sc

import json
import time

from gen_models_info import gen_object_tag

def get_symm(mesh, h_threshold=0.1, mesh_points=10000, angle_steps=20, result_matrix=True):

    # move object into initial position
    mesh.vertices -= mesh.center_mass
    # h_pit = mesh.principal_inertia_transform
    # mesh.apply_transform(h_pit)
    # mesh.show()

    mesh_sub_points = np.array(trimesh.sample.sample_surface(mesh=mesh, count=mesh_points)[0])

    # Get all rotation axes combinations
    axes = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    axis_angle = []

    for ax in axes:
        found = False
        # print(f'{ax=}')
        for i in range(angle_steps,1,-1):
            points_rot = mesh_sub_points.copy()
            rotation_matrix_h = trimesh.transformations.rotation_matrix(2 * np.pi / i, ax)
            points_rot = np.dot(points_rot, rotation_matrix_h[:3, :3].T)

            h = sc.spatial.distance.directed_hausdorff(mesh_sub_points, points_rot)[0]

            # If a symmetrie is detected break
            if h < h_threshold:
                axis_angle.append(i)
                found = True
                break

        if not found:
            axis_angle.append(1)

    if result_matrix:
        tag = gen_object_tag(axis_angle, angle_steps)
    else:
        tag = {'axis_angles': axis_angle}

    return tag
