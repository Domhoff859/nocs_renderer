import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh
import pyrender

def interpolate_color(start_color, end_color, t):
    """
    Linearly interpolates between two colors.
    """
    return [start_color[i] + t * (end_color[i] - start_color[i]) for i in range(3)]

def plot_cube(ax):
    # Define vertices of the cube
    vertices = np.array([[0, 0, 0],
                         [1, 0, 0],
                         [1, 1, 0],
                         [0, 1, 0],
                         [0, 0, 1],
                         [1, 0, 1],
                         [1, 1, 1],
                         [0, 1, 1]])

    # Define edges of the cube
    edges = [[0, 1], [1, 2], [2, 3], [3, 0],
             [4, 5], [5, 6], [6, 7], [7, 4],
             [0, 4], [1, 5], [2, 6], [3, 7]]

    # Define colors for each vertex based on NOCS principle
    vertex_colors = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
                     (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]

    # Plot edges of the cube with interpolated colors
    for edge in edges:
        start_vertex = vertices[edge[0]]
        end_vertex = vertices[edge[1]]
        start_color = vertex_colors[edge[0]]
        end_color = vertex_colors[edge[1]]
        
        # Generate points along the edge and interpolate colors
        num_points = 10
        edge_points = np.linspace(start_vertex, end_vertex, num_points)
        edge_colors = [interpolate_color(start_color, end_color, i / (num_points - 1)) for i in range(num_points)]
        
        # Plot segments along the edge with interpolated colors
        for i in range(num_points - 1):
            ax.plot3D(*zip(edge_points[i], edge_points[i+1]), color=edge_colors[i])

def convert_to_nocs(mesh):

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

# def scale_mesh(mesh):
#     """
#     Scale the mesh to fit within a unit cube.
#     """

#     max_dimension = max(mesh.bounds[1] - mesh.bounds[0])
#     scaling_factor = 1.0 / max_dimension

#     mesh.apply_scale(scaling_factor)
#     print(mesh.bounds)
#     return mesh

def scale_mesh(mesh):
    """
    Scale the mesh so that the diagonal of its tightest 3D bounding box is 1.
    """
    # Compute the bounding box
    bounding_box = mesh.bounding_box_oriented

    # Calculate the diagonal length of the bounding box
    bounds = bounding_box.bounds
    diagonal_length = np.linalg.norm(bounds[1] - bounds[0])

    # Calculate the scaling factor
    scaling_factor = 1.0 / diagonal_length

    # Apply the scaling to the mesh
    mesh.apply_scale(scaling_factor)

    print(mesh.bounds)
    return mesh

def plot_mesh(ax, mesh_path):

    # Load the mesh
    mesh = trimesh.load(mesh_path)

    mesh = scale_mesh(mesh)
    mesh = convert_to_nocs(mesh)

    target_center = np.array([0.5, 0.5, 0.5])
    current_center = np.mean(mesh.vertices, axis=0)
    center_translation = target_center - current_center

    # Apply the translation to move the center of the mesh
    mesh.apply_translation(center_translation)
    # Extract vertex colors from the PLY file
    face_colors = mesh.visual.face_colors / 255.0

    mpl_mesh = Poly3DCollection(mesh.vertices[mesh.faces], alpha=1.0)
    mpl_mesh.set_facecolor(face_colors)

    try:
        ax.add_collection3d(mpl_mesh)
        
    except Exception as e:
        print("Error loading mesh:", e)

def plot_mesh_with_pyrender(mesh_path):
    try:
        # Load the mesh
        mesh = trimesh.load(mesh_path)

        # Create a pyrender scene
        scene = pyrender.Scene()

        # Create a pyrender mesh
        mesh_node = pyrender.Mesh.from_trimesh(mesh)

        # Add the mesh to the scene
        scene.add(mesh_node)

        # Create a pyrender viewer
        viewer = pyrender.Viewer(scene, use_raymond_lighting=True)

        # Keep the viewer open
        while viewer.is_active:
            viewer.render()

    except Exception as e:
        print("Error loading mesh:", e)

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the cube
mesh_path = "models/drill.obj" 
#mesh_path = "/hdd2/real_camera_dataset/obj_models/real_test/camera_canon_len_norm.obj"
plot_mesh(ax, mesh_path)
plot_cube(ax)

# Set aspect ratio
ax.set_box_aspect([1,1,1])

# Remove grid
ax.grid(False)

# Remove axis labels
ax.set_axis_off()

# Show the plot
plt.show()

