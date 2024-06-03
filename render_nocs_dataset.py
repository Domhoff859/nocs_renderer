# parts of the code taken from:
#  https://github.com/thodan/bop_toolkit
#  https://stackoverflow.com/questions/53350391/surface-normal-calculation-from-depth-map-in-python
#  https://github.com/mmatl/pyrender/blob/master/examples/example.py

import numpy as np
import pyrender_local.pyrender as pyrender
import trimesh
import os
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from pyrender_local.pyrender import IntrinsicsCamera, SpotLight, OffscreenRenderer
from rendering import utils as renderutil
from rendering.renderer_xyz import Renderer
from rendering.model import Model3D
import random
import gc
import sys
from mpl_toolkits.mplot3d import Axes3D
import png
from PIL import Image
import json
import open3d as o3d

def plot_pcd(numpy_array):
    # Create a PointCloud object
    print(numpy_array.shape)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(numpy_array))

    o3d.visualization.draw_geometries([pcd])

# def plot_pcd(numpy_array):
#     # Reshape the array to (N, 3) where N is the total number of points (960*1280)
#     points = numpy_array.reshape(-1, 3)
    
#     # Create a PointCloud object
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points)
    
#     # Create a visualization with a coordinate frame for better context
#     vis = o3d.visualization.Visualizer()
#     vis.create_window()
#     vis.add_geometry(pcd)
    
#     # Add a coordinate frame for reference
#     vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame())
    
#     # Add a grid
#     bbox = pcd.get_axis_aligned_bounding_box()
#     grid = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(bbox)
#     vis.add_geometry(grid)
    
#     # Run the visualizer
#     vis.run()

def plot_pcd_images(xyz_estimated):
    """
    Plot point cloud from estimated XYZ values.
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    x = xyz_estimated[:, :, 0].flatten()
    y = xyz_estimated[:, :, 1].flatten()
    z = xyz_estimated[:, :, 2].flatten()

    ax.scatter(x, y, z, c='r', marker='o')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Point Cloud from RGB GT Image')

    plt.show()

def add_noise_to_depth(depth, max_noise_level=0.01):
    """
    Add Gaussian noise to a depth image.
    """
    noise_level = np.random.uniform(0, max_noise_level)
    noisy_depth = depth + np.random.normal(scale=noise_level, size=depth.shape)
    return noisy_depth

def get_surface_normal_by_depth(depth, fx=572.4114, fy=573.57043):
    """
    Calculate surface normals from depth image.
    """
    dz_dv, dz_du = np.gradient(depth)
    du_dx = fx / depth
    dv_dy = fy / depth

    dz_dx = dz_du * du_dx
    dz_dy = dz_dv * dv_dy

    normal_cross = np.dstack((-dz_dx, -dz_dy, np.ones_like(depth)))
    normal_unit = normal_cross / np.linalg.norm(normal_cross, axis=2, keepdims=True)
    normal_unit[~np.isfinite(normal_unit).all(2)] = [0, 0, 1]
    return normal_unit

def convert_to_nocs(mesh):
    """
    Convert mesh to Normalized Object Coordinate Space (NOCS).
    """
    n_vert = len(mesh.vertices)
    colors = []

    for i in range(n_vert):
        # Normalize x, y, z values to the range [-1, 1]
        r = (mesh.vertices[i, 0])
        r = (r + 1) / 2  # Map to [0, 1]

        g = (mesh.vertices[i, 1])
        g = (g + 1) / 2  # Map to [0, 1]
        
        b = (mesh.vertices[i, 2])
        b = (b + 1) / 2  # Map to [0, 1]

        # Convert to color values (0-255) and append to the colors array
        colors.append([int(r * 255), int(g * 255), int(b * 255), 255])

    new_mesh = trimesh.Trimesh(vertices=mesh.vertices,
                               faces=mesh.faces,
                               vertex_colors=colors)

    return new_mesh

def calc_cam_poses(points):
    """
    Calculate camera poses based on given points.
    """
    camera_poses = []
    for point in points:
        center = np.array([0.0, 0.0, 0.0])
        direction = center - point
        direction /= np.linalg.norm(direction)

        z_axis = np.array([0, 0, -1])
        rotation_axis = np.cross(z_axis, direction)
        rotation_angle = np.arccos(np.dot(z_axis, direction))
        rotation_matrix = trimesh.transformations.rotation_matrix(rotation_angle, rotation_axis)[:3, :3]

        camera_pose = np.eye(4)
        camera_pose[:3, :3] = rotation_matrix
        camera_pose[:3, 3] = point

        camera_poses.append(camera_pose)

    return camera_poses

def save_depth(path, im):
    """
    Save depth image as 16-bit PNG.
    """
    if not path.endswith(".png"):
        raise ValueError('Only PNG format is currently supported.')

    im = np.clip(im * 1000, 0, 65535)
    im_uint16 = np.round(im).astype(np.uint16)

    w_depth = png.Writer(im.shape[1], im.shape[0], greyscale=True, bitdepth=16)
    with open(path, 'wb') as f:
        w_depth.write(f, np.reshape(im_uint16, (-1, im.shape[1])))

def center_crop(image, crop_size):
    """
    Center crop an image to the specified size.
    """
    h, w = image.shape[:2]
    left = (w - crop_size) // 2
    top = (h - crop_size) // 2
    return image[top:top + crop_size, left:left + crop_size]

def scale_mesh(mesh):
    """
    Scale the mesh to fit within a unit cube.
    """

    max_dimension = max(mesh.bounds[1] - mesh.bounds[0])
    scaling_factor = 1.0 / max_dimension

    mesh.apply_scale(scaling_factor)
    return mesh

def save_symmetries(symmetries, path):
    list_of_lists = [matrix.tolist() for matrix in symmetries]

    with open(path, 'w') as json_file:
        json.dump(list_of_lists, json_file, indent=4)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python render_nocs_dataset.py <obj_id> <shapenet_dir> <output_dir>")
        sys.exit(1)

    # Read command line arguments
    obj_id, shapenet_dir, output_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    
    crop_size = 512
    cam_distance_factor = 4

    # Create output directory structure
    os.makedirs(output_dir, exist_ok=True)

    objs = sorted(os.listdir(shapenet_dir + "/" + obj_id))
    output_dir = output_dir + "/" + obj_id
    os.makedirs(output_dir, exist_ok=True)

    rgb_output_dir = output_dir + "/rgb"
    os.makedirs(rgb_output_dir, exist_ok=True)
    depth_output_dir = output_dir + "/depth"
    os.makedirs(depth_output_dir, exist_ok=True)
    nocs_output_dir = output_dir + "/nocs"
    os.makedirs(nocs_output_dir, exist_ok=True)
    mask_output_dir = output_dir + "/masks"
    os.makedirs(mask_output_dir, exist_ok=True)
    symmetries_output_dir = output_dir + "/symmetries"
    os.makedirs(symmetries_output_dir, exist_ok=True)

    # Camera intrinsics
    fx, fy = 572.4114 * 2, 573.57043 * 2
    cx, cy = 325.2611 * 2, 242.04899 * 2

    r = OffscreenRenderer(viewport_width=1280, viewport_height=960)
    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    # Camera intrinsics
    for idx, obj in tqdm(enumerate(objs), total=len(objs), desc="Processing Objects"):
        mesh_path = shapenet_dir + "/" + obj_id + "/" + obj + "/model.obj"
        mesh = trimesh.load(mesh_path, force='mesh')
        mesh = scale_mesh(mesh)

        #####################################################
        #
        #   your script for symmetry calculation goes here
        #
        ######################################################

        # obj_symmetries = symmetry_calculation(mesh)
        # symmetries_path = os.path.join(symmetries_output_dir, f"{idx:06d}.json")
        # save_symmetries(obj_symmetries, symmetries_path)

        ######################################################

        # Convert mesh to pyrender format
        mesh_rgb_pyrender = pyrender.Mesh.from_trimesh(mesh)

        random_intensity = np.random.uniform(1, 10, size=3)
        spot_lights = [SpotLight(color=np.ones(3), intensity=i, innerConeAngle=np.pi/16, outerConeAngle=np.pi/6) for i in random_intensity]

        cam = IntrinsicsCamera(fx, fy, cx, cy)
        n_camera = pyrender.Node(camera=cam, matrix=np.eye(4))

        scene = pyrender.Scene(ambient_light=np.array([0.2, 0.2, 0.2, 1.0]), bg_color=(0,0,0))
        scene.add_node(n_camera)
        scene.add(mesh_rgb_pyrender)

        spot_light_nodes = [scene.add(spot_light, pose=np.eye(4)) for spot_light in spot_lights]

        points = trimesh.creation.icosphere(subdivisions=2, radius=np.max(mesh.bounding_box.extents) * cam_distance_factor).vertices
        camera_poses = calc_cam_poses(points)

        #camera_poses = [camera_poses[0]]
        for i, camera_pose in enumerate(camera_poses):
            light_pose_indices = np.random.randint(len(camera_poses), size=3)    
            scene.set_pose(n_camera, pose=camera_pose)

            for j, spot_light_node in enumerate(spot_light_nodes):
                scene.set_pose(spot_light_node, pose=camera_poses[light_pose_indices[j]])

            color, depth = r.render(scene)
            rgb_path = os.path.join(rgb_output_dir, f"{idx:06d}_{i:06d}.png")
            # rgb_path = os.path.join(rgb_output_dir, obj + ".png")
            depth_path = os.path.join(depth_output_dir, f"{idx:06d}_{i:06d}.png")
            mask_path = os.path.join(mask_output_dir, f"{idx:06d}_{i:06d}.png")

            mask = r.render(scene, pyrender.RenderFlags.SEG, {node: (i + 1) for i, node in enumerate(scene.mesh_nodes)})[0]
            mask = (mask * 255).astype(np.uint8)

            color_cropped = center_crop(color, crop_size)
            mask_cropped = center_crop(mask, crop_size)
            depth_cropped = center_crop(depth, crop_size)

            cv2.imwrite(rgb_path, color_cropped[:, :, ::-1])
            save_depth(depth_path, depth_cropped)
            cv2.imwrite(mask_path, mask_cropped)

        del scene

        scene = pyrender.Scene(bg_color=(0,0,0))
        scene.add_node(n_camera)
        mesh_nocs = convert_to_nocs(mesh)
        scene.add(pyrender.Mesh.from_trimesh(mesh_nocs))

        camera_poses = calc_cam_poses(points)

        print(camera_poses)
        
        #camera_poses = [camera_poses[0]]
        for i, camera_pose in enumerate(camera_poses):

            scene.set_pose(n_camera, pose=camera_pose)

            color, _ = r.render(scene, flags=pyrender.constants.RenderFlags.FLAT | pyrender.constants.RenderFlags.DISABLE_ANTI_ALIASING)
            color_cropped = center_crop(color, crop_size)

            nocs_path = os.path.join(nocs_output_dir, f"{idx:06d}_{i:06d}.png")
            cv2.imwrite(nocs_path, color_cropped[:, :, ::-1])

            # plot_pcd_images(color)

        del scene
        gc.collect()