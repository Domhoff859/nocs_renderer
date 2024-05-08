import numpy as np
import pyrender
import trimesh
import os
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from pyrender import PerspectiveCamera,\
                     DirectionalLight, SpotLight, PointLight,\
                     MetallicRoughnessMaterial,\
                     Primitive, Mesh, Node, Scene,\
                     Viewer, OffscreenRenderer, RenderFlags, IntrinsicsCamera

from pyrender.shader_program import ShaderProgramCache

import os
import random
import gc
import sys
import imageio

def get_surface_normal_by_depth(depth, fx=572.4114 , fy=573.57043):
    """
    depth: (h, w) of float, the unit of depth is meter
    K: (3, 3) of float, the depth camere's intrinsic
    """
    # K = [[1, 0], [0, 1]] if K is None else K
    # fx, fy = K[0][0], K[1][1]

    dz_dv, dz_du = np.gradient(depth)  # u, v mean the pixel coordinate in the image
    # u*depth = fx*x + cx --> du/dx = fx / depth
    du_dx = fx / depth  # x is xyz of camera coordinate
    dv_dy = fy / depth

    dz_dx = dz_du * du_dx
    dz_dy = dz_dv * dv_dy
    # cross-product (1,0,dz_dx)X(0,1,dz_dy) = (-dz_dx, -dz_dy, 1)
    normal_cross = np.dstack((-dz_dx, -dz_dy, np.ones_like(depth)))
    # normalize to unit vector
    normal_unit = normal_cross / np.linalg.norm(normal_cross, axis=2, keepdims=True)
    # set default normal to [0, 0, 1]
    normal_unit[~np.isfinite(normal_unit).all(2)] = [0, 0, 1]
    return normal_unit

def calc_normal_image(depth_map):
    rows, cols = depth_map.shape

    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    x = x.astype(np.float32)
    y = y.astype(np.float32)

    # Calculate the partial derivatives of depth with respect to x and y
    dx = cv2.Sobel(depth_map, cv2.CV_32F, 1, 0)
    dy = cv2.Sobel(depth_map, cv2.CV_32F, 0, 1)

    # Compute the normal vector for each pixel
    normal = np.dstack((-dx, -dy, np.ones((rows, cols))))
    norm = np.sqrt(np.sum(normal**2, axis=2, keepdims=True))
    normal = np.divide(normal, norm, out=np.zeros_like(normal), where=norm != 0)

    # Map the normal vectors to the [0, 255] range and convert to uint8
    normal = (normal + 1) * 127.5
    normal = normal.clip(0, 255).astype(np.uint8)

    return normal

def convert_to_nocs(mesh):

    # x_ct = np.mean(mesh.vertices[:, 0])
    # y_ct = np.mean(mesh.vertices[:, 1])
    # z_ct = np.mean(mesh.vertices[:, 2])

    x_ct = 0
    y_ct = 0
    z_ct = 0

    x_abs = np.max(np.abs(mesh.vertices[:, 0] - x_ct))
    y_abs = np.max(np.abs(mesh.vertices[:, 1] - y_ct))
    z_abs = np.max(np.abs(mesh.vertices[:, 2] - z_ct))

    n_vert = len(mesh.vertices)
    colors = []

    for i in range(n_vert):
        # Normalize x, y, z values to the range [-1, 1]
        r = (mesh.vertices[i, 0] - x_ct) / x_abs
        r = (r + 1) / 2  # Map to [0, 1]

        g = (mesh.vertices[i, 1] - y_ct) / y_abs
        g = (g + 1) / 2  # Map to [0, 1]
        
        b = (mesh.vertices[i, 2] - z_ct) / z_abs
        b = (b + 1) / 2  # Map to [0, 1]

        # Convert to color values (0-255) and append to the colors array
        colors.append([int(r * 255), int(g * 255), int(b * 255), 255])

    new_mesh = trimesh.Trimesh(vertices=mesh.vertices,
                               faces=mesh.faces,
                               vertex_colors=colors)
    return new_mesh

def random_color_mesh(mesh):
    vis = mesh.visual.to_color()
    colors = np.random.uniform(size=(mesh.vertices.shape))

    new_mesh = trimesh.Trimesh(vertices=mesh.vertices,
                               faces=mesh.faces,
                               vertex_colors=vis.vertex_colors)

    return new_mesh

def calc_cam_poses(points):
    camera_poses = []

    for point in points:
        center = np.array([0.0, 0.0, 0.0])
        direction = center - point
        direction /= np.linalg.norm(direction)

        z_axis = np.array([0, 0, -1])
        rotation_axis = np.cross(z_axis, direction)
        rotation_angle = np.arccos(np.dot(z_axis, direction))
        rotation_matrix = trimesh.transformations.rotation_matrix(rotation_angle, rotation_axis)[:3, :3]

        # Calculate translation vector
        translation_vector = point

        # Set camera pose (rotation)
        camera_pose = np.eye(4)
        camera_pose[:3, :3] = rotation_matrix
        camera_pose[:3, 3] = translation_vector

        camera_poses.append(camera_pose)

    return camera_poses

def scale_mesh(mesh):
    # Get the bounding box of the mesh
    bbox_min, bbox_max = mesh.bounds

    # Calculate the maximum dimension of the bounding box
    max_dimension = max(bbox_max - bbox_min)

    # Calculate the scaling factor to fit the maximum dimension within the unit cube
    scaling_factor = 1.0 / max_dimension

    # Scale the mesh uniformly
    mesh = mesh.apply_scale(scaling_factor)

    return mesh

def center_crop(image, crop_size):
    h, w = image.shape[:2]
    left = (w - crop_size) // 2
    top = (h - crop_size) // 2
    right = left + crop_size
    bottom = top + crop_size
    return image[top:bottom, left:right]

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python plot_nocs_pyrender.py <obj_id> <shapenet_dir> <output_dir>")
        sys.exit(1)

    obj_id = sys.argv[1]
    shapenet_dir = sys.argv[2]
    output_dir = sys.argv[3]

    print("output_dir: ", output_dir)
    print("obj_id: ", obj_id)
    print()

    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    os.makedirs(output_dir, exist_ok=True)

    objs = sorted(os.listdir(shapenet_dir + "/" + obj_id))
    output_dir = output_dir + "/" + obj_id
    os.makedirs(output_dir, exist_ok=True)

    rgb_output_dir = output_dir + "/rgb"
    os.makedirs(rgb_output_dir, exist_ok=True)
    nocs_output_dir = output_dir + "/nocs"
    os.makedirs(nocs_output_dir, exist_ok=True)
    # depth_output_dir = output_dir + "/depth"
    # os.makedirs(depth_output_dir, exist_ok=True)
    normal_output_dir = output_dir + "/normals"
    os.makedirs(normal_output_dir, exist_ok=True)
    mask_output_dir = output_dir + "/masks"
    os.makedirs(mask_output_dir, exist_ok=True)

    r = OffscreenRenderer(viewport_width=640, viewport_height=480)
    r_normals = OffscreenRenderer(viewport_width=640, viewport_height=480)
    r_normals._renderer._program_cache = ShaderProgramCache(shader_dir="shaders")

    for idx, obj in tqdm(enumerate(objs), total=len(objs), desc="Processing Objects"):
        mesh_path = shapenet_dir + "/" + obj_id + "/" + obj + "/model.obj"

        mesh = trimesh.load(mesh_path, force='mesh')
        #mesh.vertices -= mesh.center_mass

        mesh_rgb_pyrender = pyrender.Mesh.from_trimesh(mesh)

        random_intensity = np.random.uniform(1, 10, size=3)
        intensity_1, intensity_2, intensity_3 = random_intensity

        spot_l1 = SpotLight(color=np.ones(3), intensity=intensity_1,
                        innerConeAngle=np.pi/16, outerConeAngle=np.pi/6)
        spot_l2 = SpotLight(color=np.ones(3), intensity=intensity_2,
                        innerConeAngle=np.pi/16, outerConeAngle=np.pi/6)
        spot_l3 = SpotLight(color=np.ones(3), intensity=intensity_3,
                        innerConeAngle=np.pi/16, outerConeAngle=np.pi/6)

        #cam = PerspectiveCamera(yfov=(np.pi / 3.0))
        cam = IntrinsicsCamera(fx=572.4114 , fy=573.57043 , cx=325.2611, cy=242.04899)
        n_camera = pyrender.Node(camera=cam, matrix=np.eye(4))

        scene = pyrender.Scene(ambient_light=np.array([0.1, 0.1, 0.1, 1.0]), bg_color=(1,1,1))

        scene.add_node(n_camera)
        scene.add(mesh_rgb_pyrender)

        spot_l1_node = scene.add(spot_l1, pose=np.eye(4))
        spot_l2_node = scene.add(spot_l2, pose=np.eye(4))
        spot_l3_node = scene.add(spot_l3, pose=np.eye(4))

        # Generate camera poses using icosphere sampling
        points = trimesh.creation.icosphere(subdivisions=2, radius=np.max(mesh.bounding_box.extents) * 3).vertices
        camera_poses = calc_cam_poses(points)

        for i, camera_pose in enumerate(camera_poses):
            
            light_pose_indices = np.random.randint(len(camera_poses), size=3)    
            idx1, idx2, idx3 = light_pose_indices      

            scene.set_pose(n_camera, pose=camera_pose)
            scene.set_pose(spot_l1_node, pose=camera_poses[idx1])
            scene.set_pose(spot_l2_node, pose=camera_poses[idx2])
            scene.set_pose(spot_l3_node, pose=camera_poses[idx3])

            # Render the scene
            color, depth = r.render(scene)
            rgb_path = os.path.join(rgb_output_dir, f"{idx:06d}_{i:06d}.png")
            #depth_path = os.path.join(depth_output_dir, f"{idx:06d}_{i:06d}.png")
            mask_path = os.path.join(mask_output_dir, f"{idx:06d}_{i:06d}.png")
            normal_path = os.path.join(normal_output_dir, f"{idx:06d}_{i:06d}.png")

            normals = get_surface_normal_by_depth(depth, fx=572.4114 , fy=573.57043)
            normals = np.uint8((normals + 1) / 2 * 255)

            print("depth np.min: ", np.min(depth))
            print("depth np.max: ", np.max(depth))

            nm = {node: (i + 1) for i, node in enumerate(scene.mesh_nodes)}
            mask = r.render(scene, RenderFlags.SEG, nm)[0]
            mask = (mask * 255).astype(np.uint8)

            color_cropped = center_crop(color, 256)
            normals_cropped = center_crop(normals, 256)
            mask_cropped = center_crop(mask, 256)

            cv2.imwrite(rgb_path, color_cropped[:, :, ::-1])
            #cv2.imwrite(depth_path, depth)
            cv2.imwrite(mask_path, mask_cropped)
            cv2.imwrite(normal_path, normals_cropped[:, :, ::-1])

        del scene
        gc.collect()

        # mesh_rgb_pyrender = pyrender.Mesh.from_trimesh(mesh, smooth = False)
        # scene = pyrender.Scene(ambient_light=np.array([1,1,1, 1.0]), bg_color=(1,1,1))
        # scene.add_node(n_camera)
        # scene.add(mesh_rgb_pyrender)

        # for i, camera_pose in enumerate(camera_poses):

        #     scene.set_pose(n_camera, pose=camera_pose)

        #     # Render the scene
        #     normals, depth = r_normals.render(scene)

        #     normal_path = os.path.join(normal_output_dir, f"{idx:06d}_{i:06d}.png")

        #     cv2.imwrite(normal_path, normals[:, :, ::-1])

        # del scene
        # gc.collect()

        scene = pyrender.Scene(ambient_light=np.array([1,1,1, 1.0]), bg_color=(1,1,1))
        scene.add_node(n_camera)
        mesh_nocs = scale_mesh(mesh)
        mesh_nocs = convert_to_nocs(mesh_nocs)

        mesh_nocs_pyrender = pyrender.Mesh.from_trimesh(mesh_nocs)
        scene.add(mesh_nocs_pyrender)

        points = trimesh.creation.icosphere(subdivisions=2, radius=np.max(mesh_nocs.bounding_box.extents) * 3).vertices
        camera_poses = calc_cam_poses(points)

        for i, camera_pose in enumerate(camera_poses):

            scene.set_pose(n_camera, pose=camera_pose)

            color, _ = r.render(scene)
            color_cropped = center_crop(color, 256)

            nocs_path = os.path.join(nocs_output_dir, f"{idx:06d}_{i:06d}.png")
            cv2.imwrite(nocs_path, color_cropped[:, :, ::-1])

        del scene
        gc.collect()