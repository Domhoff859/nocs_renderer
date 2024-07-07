# parts of the code taken from:
#  https://github.com/thodan/bop_toolkit
#  https://stackoverflow.com/questions/53350391/surface-normal-calculation-from-depth-map-in-python
#  https://github.com/mmatl/pyrender/blob/master/examples/example.py

import os
import sys

import numpy as np
import trimesh
import cv2
from tqdm import tqdm
import png
import json
import open3d as o3d

import bop_io
import pyrender_local.pyrender as pyrender
from pyrender_local.pyrender import IntrinsicsCamera, SpotLight, OffscreenRenderer
from star_dash import StarRepresentation, DashRepresentation

ROOT_DIR = os.path.abspath(".")
sys.path.append(ROOT_DIR) 
sys.path.append("./bop_toolkit")
from bop_toolkit_lib import inout, dataset_params

def plot_pcd(numpy_array):
    # Create a PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(numpy_array))

    o3d.visualization.draw_geometries([pcd])

def convert_to_nocs(mesh):
    """
    Convert mesh to Normalized Object Coordinate Space (NOCS).
    """
    n_vert = len(mesh.vertices)
    colors = []

    original_vertices = np.copy(mesh.vertices)
    original_faces = np.copy(mesh.faces)

    mesh = scale_mesh(mesh)

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

    new_mesh = trimesh.Trimesh(vertices=original_vertices,
                               faces=original_faces,
                               vertex_colors=colors)

    return new_mesh

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

def scale_mesh(mesh):
    """
    Scale the mesh to fit within a unit cube.
    """

    max_dimension = np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])
    scaling_factor = 1.0 / max_dimension

    mesh.apply_scale(scaling_factor)

    return mesh

def crop_and_resize(img, bbox, target_size=128):
    # Calculate the bounding box dimensions and center
    bbox_width = bbox[3] - bbox[1]
    bbox_height = bbox[2] - bbox[0]
    center_x = (bbox[1] + bbox[3]) // 2
    center_y = (bbox[0] + bbox[2]) // 2
    
    # Enlarge the bounding box
    enlarged_size = int(max(bbox_width, bbox_height) * 1.5)
    # enlarged_size = 128
    
    crop_xmin = max(center_x - enlarged_size // 2, 0)
    crop_xmax = min(center_x + enlarged_size // 2, img.shape[1])
    crop_ymin = max(center_y - enlarged_size // 2, 0)
    crop_ymax = min(center_y + enlarged_size // 2, img.shape[0])
    
    # Crop and pad the image and img_r
    if img.ndim == 3:
        cropped_img = np.zeros((enlarged_size, enlarged_size, img.shape[2]), dtype=img.dtype)
    else:
        cropped_img = np.zeros((enlarged_size, enlarged_size), dtype=img.dtype)
    
    y_offset = (enlarged_size - (crop_ymax - crop_ymin)) // 2
    x_offset = (enlarged_size - (crop_xmax - crop_xmin)) // 2
    
    if img.ndim == 3:
        cropped_img[y_offset:y_offset + (crop_ymax - crop_ymin), x_offset:x_offset + (crop_xmax - crop_xmin)] = img[crop_ymin:crop_ymax, crop_xmin:crop_xmax, :]
    else:
        cropped_img[y_offset:y_offset + (crop_ymax - crop_ymin), x_offset:x_offset + (crop_xmax - crop_xmin)] = img[crop_ymin:crop_ymax, crop_xmin:crop_xmax]
    
    # Resize if necessary
    if cropped_img.shape[0] != target_size:
        scale_factor = target_size / float(cropped_img.shape[0])
        if cropped_img.dtype != np.uint8:
            cropped_img = cropped_img.astype(np.uint8)
        cropped_img = cv2.resize(cropped_img, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
        
    
    return cropped_img

def pyrender_cam_pose():

    # Identity matrix
    camera_pose = np.eye(4)

    # Rotate 180 degrees around the Y-axis
    rotation_y_180 = np.array([
        [-1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])

    # Rotate 180 degrees around the Z-axis
    rotation_z_180 = np.array([
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    # Combine rotations: first Y, then Z
    camera_pose = np.dot(rotation_z_180, rotation_y_180)

    return camera_pose

def load_star_dash_model_info(model_info: dict) -> dict:
    """
    Refactor the model info as needed by the Star and Dash Representation

    Args:
        model_info (dict): BOP model info

    Returns:
        dict: StarDash model info
    """
    star_dash_model_info = {}
    for object_id in model_info:
        object_info = {}
        
        key = str(object_id)
        assert(key in model_info)

        # Extract the relevant information for the object
        object_info["diameter"] = model_info[key]["diameter"]
        object_info["mins"] = np.array([model_info[key]["min_x"],model_info[key]["min_y"],model_info[key]["min_z"]])
        object_info["maxs"] = np.array([model_info[key]["size_x"],model_info[key]["size_y"],model_info[key]["size_z"]]) +  object_info["mins"]

        # Check if discrete symmetries are present
        if "symmetries_discrete" in model_info[key]:
            object_info["symmetries_discrete"] = [np.array(_).reshape((4,4)) for _ in model_info[key]["symmetries_discrete"]]
        else:
            object_info["symmetries_discrete"] = []

        # Check if continuous symmetries are present
        object_info["symmetries_continuous"] = "symmetries_continuous" in model_info[key]
            
        star_dash_model_info[object_id] = object_info
        
    return star_dash_model_info

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python render_bop_dataset.py <bop_directory> <dataset>")
        sys.exit(1)

    # smallest tolerable visibilty of object for rendering
    visibility_threshold = 0.2

    # Read command line arguments
    bop_directory, dataset = sys.argv[1], sys.argv[2]

    # dataset infos
    bop_dir, source_dir, model_plys, model_info, objs, rgb_files,\
        depth_files, mask_files, mask_visib_files, gts, cam_param_global, scene_cam =\
             bop_io.get_dataset(bop_directory, dataset, incl_param=True)
    rgb_fn = rgb_files[0]
    print()
    
    

    # Create output directory structure
    output_dir = bop_directory + "/" + dataset + "/xyz_data"
    os.makedirs(output_dir, exist_ok=True)

    # initialize pyrender renderer
    r = OffscreenRenderer(viewport_width=640, viewport_height=480)

    # StarDash library hier einbetten
    star_dash_model_info = load_star_dash_model_info(model_info=model_info)
    
    star = StarRepresentation(model_info=star_dash_model_info)
    dash = DashRepresentation(model_info=star_dash_model_info)

    # Camera intrinsics
    for idx, obj in enumerate(objs):
        # """
        # Only train objects:
        # inf : 1, 4 
        # 1 Fold: 21, 22
        # 2 Fold: 5, 9,  
        # 4 Fold: 27
        # """
        if obj not in [1, 4, 5, 9, 21, 22, 27]:
            continue
        
        
        print(f"Processing object {obj}...")
        
        # create output folders
        obj_output_dir = output_dir + "/" + str(obj)
        os.makedirs(obj_output_dir, exist_ok=True)

        rgb_output_dir = obj_output_dir + "/rgb"
        os.makedirs(rgb_output_dir, exist_ok=True)
        nocs_output_dir = obj_output_dir + "/nocs"
        os.makedirs(nocs_output_dir, exist_ok=True)
        star_output_dir = obj_output_dir + "/star"
        os.makedirs(star_output_dir, exist_ok=True)
        dash_output_dir = obj_output_dir + "/dash"
        os.makedirs(dash_output_dir, exist_ok=True)
        mask_output_dir = obj_output_dir + "/mask"
        os.makedirs(mask_output_dir, exist_ok=True)
        mask_visib_output_dir = obj_output_dir + "/mask_visib"
        os.makedirs(mask_visib_output_dir, exist_ok=True)
        cam_R_m2c_output_dir = obj_output_dir + "/cam_R_m2c"
        os.makedirs(cam_R_m2c_output_dir, exist_ok=True)
        
        # load mesh, millimeters to meters and NOCS mesh calculation
        mesh_path = model_plys[idx]
        mesh = trimesh.load(mesh_path, force='mesh')
        # millimeters to meters
        mesh.vertices = mesh.vertices / 1000
        # create NOCS mesh from normal mesh
        mesh_nocs = convert_to_nocs(mesh)
        mesh_nocs_pyrender = pyrender.Mesh.from_trimesh(mesh_nocs)

        # iterate through all RGB files
        for img_id in tqdm(range(len(rgb_files)), total=len(rgb_files), desc="Processing Images"):
            gt_img = gts[img_id]

            fx = scene_cam[img_id]["cam_K"][0][0]
            fy = scene_cam[img_id]["cam_K"][1][1]
            cx = scene_cam[img_id]["cam_K"][0][2]
            cy = scene_cam[img_id]["cam_K"][1][2]

            # create scene camera
            cam = IntrinsicsCamera(fx, fy, cx, cy)
            n_camera = pyrender.Node(camera=cam, matrix=pyrender_cam_pose())

            # iterate through the instances (=objects) in each RGB file
            for gt_instance, gt in enumerate(gt_img):            
                obj_id = int(gt['obj_id'])
                # if the instance is not the object we are currently looking for -> skip!
                if obj_id != int(obj):
                    continue   

                # create blank scene
                scene = pyrender.Scene(bg_color=(0,0,0))

                # create pose matrix
                pose = np.eye(4)
                pose[:3, 3] = np.array((gt['cam_t_m2c'])/1000)[:,0]
                pose[:3, :3] = np.array(gt['cam_R_m2c']).reshape(3,3)

                # add mesh and camera to the blank scene
                scene.add(mesh_nocs_pyrender, pose=pose)
                scene.add_node(n_camera)

                rgb_fn = rgb_files[img_id]
                depth_fn = depth_files[img_id]

                # for now this is very ugly code
                string_without_extension = rgb_fn[:-4]
                img_string = string_without_extension[-6:]
                scene_string = string_without_extension[-17:-11]

                for index, item in enumerate(gts[img_id]):
                    if item['obj_id'] == obj_id:
                        break

                parts_mask_files= mask_files[img_id].rsplit('_', 1)
                parts_mask_visib_files= mask_visib_files[img_id].rsplit('_', 1)
                new_last_part = f"_{index:06d}.png"
                new_path_mask_files = f"{parts_mask_files[0]}{new_last_part}"
                new_path_mask_visib_files = f"{parts_mask_visib_files[0]}{new_last_part}"

                mask = inout.load_im(new_path_mask_files)>0     
                mask_visib = inout.load_im(new_path_mask_visib_files)>0

                img_r, depth_rend = r.render(scene, flags=pyrender.constants.RenderFlags.FLAT | pyrender.constants.RenderFlags.DISABLE_ANTI_ALIASING)
                img = inout.load_im(rgb_fn)
                
                ################
                #
                # put the STAR and DASH map calculation here
                # 
                ###############
                
                # =================================================================================
                valid_star = star.calculate(po_image=img_r[np.newaxis, :, :, :], object_id=str(obj))
                valid_dash = dash.calculate(R=np.array(gt['cam_R_m2c'])[np.newaxis, :, :], po_image=img_r[np.newaxis, :, :, :], object_id=str(obj))
                
                
                # Normalize the STAR and DASH maps
                valid_star = valid_star[0, :, : ,:]
                valid_star = np.where(valid_star != 0, valid_star / np.sqrt(2) / 2 + 127.5, 0)
                valid_star = np.array(valid_star, dtype=np.uint8)
                
                valid_dash = valid_dash[0, :, :, :]
                valid_dash = np.where(valid_dash != 0, valid_dash / 2 + 127.5, 0)
                valid_dash = np.array(valid_dash, dtype=np.uint8)
                # =================================================================================

                vu_valid = np.where(depth_rend > 0)

                if vu_valid[0].size == 0 or vu_valid[1].size == 0:
                    bbox_gt = np.zeros(4, dtype=int)
                else:
                    bbox_gt = np.array([np.min(vu_valid[0]), np.min(vu_valid[1]), np.max(vu_valid[0]), np.max(vu_valid[1])])

                x_min, y_min, x_max, y_max = bbox_gt

                # Slice the masks to only include the area within the bounding box
                mask_bbox = mask[x_min:x_max, y_min:y_max]
                mask_visib_bbox = mask_visib[x_min:x_max, y_min:y_max]
                
                # Calculate the number of non-zero pixels in the visibility mask within the bounding box
                visible_pixels_bbox = np.count_nonzero(mask_visib_bbox)

                # Calculate the number of non-zero pixels in the total mask within the bounding box
                pixels_bbox = np.count_nonzero(mask_bbox)

                # Calculate the visibility percentage within the bounding box
                if pixels_bbox > 0:
                    visibility_percentage_bbox = float(visible_pixels_bbox) / float(pixels_bbox)
                else:
                    visibility_percentage_bbox = 0.0

                # print("Visibility percentage within bounding box: ", visibility_percentage_bbox)
                
                rgb_data = crop_and_resize(img, bbox_gt)
                xyz_data = crop_and_resize(img_r, bbox_gt)
                star_data = crop_and_resize(valid_star, bbox_gt)
                dash_data = crop_and_resize(valid_dash, bbox_gt)

                cropped_mask = crop_and_resize(mask, bbox_gt)
                cropped_mask_visib = crop_and_resize(mask_visib, bbox_gt)

                if cropped_mask.dtype != np.uint8:
                    cropped_mask = cropped_mask.astype(np.uint8)
                if cropped_mask_visib.dtype != np.uint8:
                    cropped_mask_visib = cropped_mask_visib.astype(np.uint8)
                if (
                    rgb_data.shape[0] > 20 and rgb_data.shape[1] > 20
                    and xyz_data.shape[0] > 20 and xyz_data.shape[1] > 20
                    and visibility_percentage_bbox > visibility_threshold
                ):
                    # Check if cropped_mask is a binary mask and convert to np.uint8
                    if len(np.unique(cropped_mask)) <= 2:
                        cropped_mask = cropped_mask * 255
                    if len(np.unique(cropped_mask_visib)) <= 2:
                        cropped_mask_visib = cropped_mask_visib * 255

                    xyz_sub_fn = os.path.join(nocs_output_dir, f"{scene_string}_{img_string}_{gt_instance:06d}.png")
                    rgb_sub_fn = os.path.join(rgb_output_dir, f"{scene_string}_{img_string}_{gt_instance:06d}.png")
                    dash_sub_fn = os.path.join(dash_output_dir, f"{scene_string}_{img_string}_{gt_instance:06d}.png")
                    star_sub_fn = os.path.join(star_output_dir, f"{scene_string}_{img_string}_{gt_instance:06d}.png")
                    mask_sub_fn = os.path.join(mask_output_dir, f"{scene_string}_{img_string}_{gt_instance:06d}.png")
                    mask_visib_sub_fn = os.path.join(mask_visib_output_dir, f"{scene_string}_{img_string}_{gt_instance:06d}.png")
                    cam_R_m2c_sub_fn = os.path.join(cam_R_m2c_output_dir, f"{scene_string}_{img_string}_{gt_instance:06d}.npy")

                    # mask_sub_fn = os.path.join(mask_sub_dir, f"{scene_string}_{img_string}_{gt_instance:06d}.png")
                    # mask_visib_sub_fn = os.path.join(mask_visib_sub_dir, f"{scene_string}_{img_string}_{gt_instance:06d}.png")
                    
                    cv2.imwrite(xyz_sub_fn, xyz_data[:, :, ::-1])
                    cv2.imwrite(rgb_sub_fn, rgb_data[:, :, ::-1])
                    cv2.imwrite(dash_sub_fn, dash_data)
                    cv2.imwrite(star_sub_fn, star_data)
                    cv2.imwrite(mask_sub_fn, cropped_mask)
                    cv2.imwrite(mask_visib_sub_fn, cropped_mask_visib)
                    np.save(cam_R_m2c_sub_fn, np.array(gt['cam_R_m2c']).reshape(3,3))

                del scene

